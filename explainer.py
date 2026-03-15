import torch
import hashlib
import numpy as np
from PIL import Image
from math import sqrt
import torch.nn as nn
import networkx as nx
from io import BytesIO
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph

EPS = 1e-15 # Epsilon
SEED = 13

class IlluminatiExplainer(nn.Module):

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'mask_l1': 0.001,   # L1 regularization coefficient for sparsity
        'mask_var': 0.001,  # Variance regularization coefficient to encourage spread
    }

    def __init__(self, model, epochs: int = 50, lr: float = 0.01,
            agg1="max", agg2="max", num_hops: int = None, device: str = "cpu", seed: int = SEED):
        super(IlluminatiExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.device = device
        self.seed = seed
        # self.drop = nn.Dropout(p=0.2)
        self.drop = nn.Identity()
        self.model.to(device)

        # Additional learnable scaling parameters for the mask scores:
        self.temp_edge = nn.Parameter(torch.tensor(1.0))   # Temperature scaling for edge masks
        self.temp_node = nn.Parameter(torch.tensor(1.0))   # Temperature scaling for node feature masks
        self.power_edge = nn.Parameter(torch.tensor(1.0))  # Power transform for edge masks
        self.power_node = nn.Parameter(torch.tensor(1.0))  # Power transform for node masks

        # Define aggregation functions
        if agg1 == "mean":
            self.agg1 = torch.mean
        elif agg1 == "min":
            self.agg1 = torch.min
        elif agg1 == "max":
            self.agg1 = torch.max
        elif agg1 == "sum":
            self.agg1 = torch.sum
        else:
            self.agg1 = self.custom_agg  # use custom if no standard option fits

        if agg2 == "mean":
            self.agg2 = torch.mean
        elif agg2 == "min":
            self.agg2 = torch.min
        elif agg2 == "max":
            self.agg2 = torch.max
        elif agg2 == "sum":
            self.agg2 = torch.sum
        else:
            self.agg2 = self.custom_agg

    def __loss__(self, logits, pred_label, loss_fc=None):
        """ 
            Enhanced loss function to include L1 regularization (to enforce sparsity) and a variance regularization term (to encourage a broader spread in mask values).
        """
        if loss_fc is None:
            loss_fc = F.cross_entropy
            if len(logits.shape) == 1 and logits.shape[0] == 1:
                loss_fc = F.binary_cross_entropy_with_logits
                pred_label = pred_label.float()
        pred_label = torch.abs(pred_label)
        loss = loss_fc(logits, pred_label)

        # Scale the raw masks using our adaptive scaling function
        m_edge = self.scale_mask(self.edge_mask, mask_type='edge')
        m_node = self.scale_mask(self.node_feat_mask, mask_type='node')

        # Original regularization for edge masks
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m_edge)
        ent_edge = -m_edge * torch.log(m_edge + EPS) - (1 - m_edge) * torch.log(1 - m_edge + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent_edge.mean()

        # Regularization for node feature masks
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m_node)
        ent_node = -m_node * torch.log(m_node + EPS) - (1 - m_node) * torch.log(1 - m_node + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent_node.mean()

        # L1 regularization to enforce sparsity (make masks sparser)
        loss = loss + self.coeffs['mask_l1'] * (torch.norm(m_edge, 1) + torch.norm(m_node, 1))

        # Variance regularization to encourage mask scores to be spread out (less uniform)
        loss = loss - self.coeffs['mask_var'] * (m_edge.var() + m_node.var())

        return loss

    def __get_indices__(self, data):
        (N, F), E = data.x.size(), data.edge_index.size(1)

        self.out_edge_mask = torch.zeros(N, E, dtype=torch.bool)
        self.in_edge_mask = torch.zeros(N, E, dtype=torch.bool)
        for n in range(N):
            self.out_edge_mask[n] = (data.edge_index[0] == n) & (data.edge_index[1] != n)
            self.in_edge_mask[n] = (data.edge_index[1] == n) & (data.edge_index[0] != n)

        self.out_degree = torch.zeros(N, dtype=torch.int)
        self.in_degree = torch.zeros(N, dtype=torch.int)
        for n in range(N):
            in_num = torch.sum(self.in_edge_mask[n], dtype=torch.int)
            out_num = torch.sum(self.out_edge_mask[n], dtype=torch.int)
            self.in_degree[n] = in_num
            self.out_degree[n] = out_num

        self.self_loop_mask = torch.zeros(N, dtype=torch.long)
        for e in range(E):
            if data.edge_index[0, e] == data.edge_index[1, e]:
                self.self_loop_mask[data.edge_index[0, e]] = e
        if self.self_loop_mask.sum() == 0:
            self.self_loop_mask = None

    def __set_masks__(self, data, node: bool = True, synchronize: bool = True, edge_mask=None):
        # Set the seed for reproducibility
        self.__set_seed__(data)
        (N, F), E = data.x.size(), data.edge_index.size(1)
        num_nodes = N

        if edge_mask is not None:
            for module in self.model.modules():
                if isinstance(module, MessagePassing):
                    module.__explain__ = True
                    module.__edge_mask__ = edge_mask
            return

        std = 0.1
        node_feat_mask = torch.randn(1, F, generator=self.gen, device=self.device) * std if not node else \
                 torch.randn(N, F, generator=self.gen, device=self.device) * std
        edge_mask = torch.randn(E, generator=self.gen, device=self.device) * std
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        edge_mask = torch.randn(E, generator=self.gen, device=self.device) * std
        if not node and self.self_loop_mask is not None:
            edge_mask[self.self_loop_mask] = torch.ones(num_nodes)
        if node and synchronize:
            node_feat_mask = torch.mean(node_feat_mask, dim=-1, keepdim=True)
        self.node_feat_mask = nn.Parameter(node_feat_mask)
        self.edge_mask = edge_mask

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
        self.out_degree = None
        self.in_degree = None
        self.out_edge_mask = None
        self.in_edge_mask = None
        self.self_loop_mask = None
        self.node_feat_mask = None
        self.edge_mask = None

    def __refine_mask__(self, mask, beta=1., training=True):
        if training:
            random_noise = torch.rand(mask.shape, generator=self.gen, device=mask.device)
            random_noise = torch.log(random_noise) - torch.log(1 - random_noise)
            s = (random_noise + mask) / beta
            s = s.sigmoid()
            z = s * 1.5 - 0.25
            z = z.clamp(0, 1)
        else:
            z = (mask / beta).sigmoid()
        return z
    
    def __set_seed__(self, data):
        # Hash the input graph
        x_bytes = data.x.detach().cpu().numpy().tobytes()
        x_hash = hashlib.md5(x_bytes).digest()

        # Hash the model parameters
        params = torch.cat([p.detach().flatten().cpu() for p in self.model.parameters()])
        model_hash = hashlib.md5(params.numpy().tobytes()).digest()

        # Combine both
        combined = hashlib.md5(x_hash + model_hash).hexdigest()
        graph_model_seed = int(combined[:8], 16)

        self.gen = torch.Generator(device=self.device).manual_seed(graph_model_seed)

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__
        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k
            
    def custom_agg(self, mask):
        # Custom aggregation function, similar to the original 'agg' function.
        mask = mask.view(-1)
        eps = 1e-6

        print(mask.numel())

        return torch.prod(mask + eps)**(1.0 / mask.numel())
    
    def scale_mask(self, mask, mask_type='edge', eps = 1e-6):
        """
            This function normalizes the raw mask values and applies the learnable temperature and power transforms. 
            The final output is squashed to the [0,1] range using a sigmoid.
        """
        # Normalize mask values (subtract mean, divide by std) to amplify differences.
        mean_val = mask.mean()
        std_val = mask.std() + eps
        normalized = (mask - mean_val) / std_val

        # Apply the learnable temperature and power scaling
        if mask_type == 'edge':
            scaled = (normalized * self.temp_edge).pow(self.power_edge)
        else:
            scaled = (normalized * self.temp_node).pow(self.power_node)

        # Return mask scores in the [0, 1] range via sigmoid
        return torch.sigmoid(scaled)
    
    def integrated_gradients(self, data, target_idx, baseline=None, steps=50):
        """
        Compute integrated gradients for the node features for a given target index.
        """
        if baseline is None:
            baseline = torch.zeros_like(data.x)
        # Generate scaled inputs along the path from baseline to the input
        scaled_inputs = [baseline + float(i) / steps * (data.x - baseline) for i in range(steps + 1)]
        grads = []
        for x in scaled_inputs:
            x.requires_grad = True
            out = self.model(x)
            target = out[target_idx]
            self.model.zero_grad()
            target.backward(retain_graph=True)
            grads.append(x.grad.detach())
        avg_grads = torch.mean(torch.stack(grads), dim=0)
        integrated_grads = (data.x - baseline) * avg_grads

        return integrated_grads
    
    def plot_node_scores_distribution(self, node_scores, bins=30):
        """
        Create and return a histogram of the node scores as a PIL Image object.
        """
        # Convert tensor to NumPy array.
        node_scores_np = node_scores.detach().cpu().numpy()
        nonzero_scores_np = node_scores_np[np.nonzero(node_scores_np)]
        
        # Create the plot.
        plt.figure()
        plt.hist(nonzero_scores_np, bins=bins)
        plt.title("Distribution of Node Scores")
        plt.xlabel("Score")
        plt.ylabel("Count")
        
        # Save the plot to an in-memory buffer.
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()  # Close the plot to free up memory.
        buf.seek(0)
        
        # Return the image object created from the in-memory buffer.
        return Image.open(buf)
    
    def explain_graph(self, data, loss_fc, node: bool = True, synchronize: bool = False):
        self.model.eval()
        self.__clear_masks__()
        (N, F), E = data.x.size(), data.edge_index.size(1)

        # Get prediction WITHOUT torch.no_grad() to allow gradients
        out = self.model(data.to(self.device))
        pred_label = out.detach()  # Retain real-valued prediction for explanation

        self.__get_indices__(data)
        self.__set_masks__(data, node=node, synchronize=synchronize)
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            node_feat_mask = self.__refine_mask__(self.node_feat_mask, beta=(epoch + 1) / self.epochs) if node else self.node_feat_mask.sigmoid()
            h = data.x * node_feat_mask
            data_tmp = Data(x=h, edge_index=data.edge_index, batch=torch.zeros(N, dtype=torch.long))
            edge_mask = self.__refine_mask__(self.edge_mask, beta=(epoch + 1) / self.epochs) if node else self.edge_mask.sigmoid()
            self.__set_masks__(data_tmp, edge_mask=edge_mask)
            out = self.model(data_tmp.to(self.device))
            loss = self.__loss__(out, pred_label, loss_fc)
            loss.backward()
            optimizer.step()

        node_feat_mask = self.__refine_mask__(self.node_feat_mask, training=False) if node else self.node_feat_mask.sigmoid()
        edge_mask = self.__refine_mask__(self.edge_mask, training=False) if node else self.edge_mask.sigmoid()
        node_mask = torch.zeros(node_feat_mask.shape[0])

        if node:
            node_feat_msg = torch.sum(node_feat_mask * data.x, dim=-1).view(-1)
            x = data.x.clone()
            x[x > 0.] = 1.
            node_feat_mask = node_feat_mask * x
            for n in range(N):
                idx = torch.nonzero(x[n])
                node_feat_msg[n] = self.custom_agg(node_feat_mask[n, idx])
            for n in range(N):
                if self.out_degree[n] > 0 or self.in_degree[n] > 0:
                    out_masks = torch.zeros(1)
                    if self.out_degree[n] > 0:
                        out_masks = edge_mask[self.out_edge_mask[n]]
                    node_mask_out = out_masks * node_feat_msg[n]
                    node_mask_out = self.agg1(node_mask_out)
                    in_masks = edge_mask[self.self_loop_mask[n]] if self.self_loop_mask is not None else torch.zeros(1)
                    node_mask_in = in_masks * node_feat_msg[n]
                    if self.in_degree[n] > 0:
                        in_nodes = data.edge_index[0, self.in_edge_mask[n]]
                        in_masks = edge_mask[self.in_edge_mask[n]]
                        if self.self_loop_mask is not None:
                            in_masks = torch.cat((in_masks.view(-1), edge_mask[self.self_loop_mask[n]].view(-1)))
                        node_mask_in = in_masks * node_feat_msg[in_nodes]
                    node_mask_in = self.agg1(node_mask_in)
                    node_mask[n] = self.agg2(torch.cat((node_mask_in.view(-1), node_mask_out.view(-1))))
        else:
            node_mask = torch.zeros(N)
            for n in range(N):
                out_max = torch.tensor(0, dtype=torch.float)
                in_max = torch.tensor(0, dtype=torch.float)
                if self.out_degree[n] > 0:
                    out_max = torch.max(edge_mask[self.out_edge_mask[n]])
                if self.in_degree[n] > 0:
                    in_max = torch.max(edge_mask[self.in_edge_mask[n]])
                node_mask[n] = torch.max(out_max, in_max)

        node_mask = self.scores_scaling_transform(node_mask)
        self.__clear_masks__()
        return node_feat_mask, edge_mask, node_mask

    def explain_node(self, data, loss_fc, idx: int = 0, node: bool = True, synchronize: bool = False):
        self.model.eval()
        self.__clear_masks__()

        node_idx, edge_idx, node_map, edge_map = k_hop_subgraph(idx, self.num_hops, data.edge_index, relabel_nodes=True)
        idx_sub = node_map[0]
        data_sub = Data(x=data.x[node_idx], edge_index=edge_idx, y=data.y[node_idx])
        (N, F), E = data_sub.x.size(), data_sub.edge_index.size(1)

        with torch.no_grad():
            data_sub.to(self.device)
            out = self.model(data_sub.x)[idx_sub]
            data_sub.cpu()
            out.cpu()
            if len(out.shape) == 1 and out.shape[0] == 1:
                pred_label = torch.round(out)
            else:
                pred_label = out.argmax(dim=-1)

        self.__get_indices__(data_sub)
        self.__set_masks__(data_sub, node=node, synchronize=synchronize)
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            node_feat_mask = self.__refine_mask__(self.node_feat_mask, beta=(epoch + 1) / self.epochs) if node else self.node_feat_mask.sigmoid()
            h = data_sub.x * node_feat_mask
            edge_mask = self.__refine_mask__(self.edge_mask, beta=(epoch + 1) / self.epochs) if node else self.edge_mask.sigmoid()
            self.__set_masks__(data_sub, edge_mask=edge_mask)
            out = self.model(h, data_sub.edge_index)[idx_sub].unsqueeze(0)
            loss = self.__loss__(out, pred_label.unsqueeze(0), loss_fc)
            loss.backward()
            optimizer.step()

        node_feat_mask = self.__refine_mask__(self.node_feat_mask, training=False) if node else self.node_feat_mask.sigmoid()
        edge_mask = self.__refine_mask__(self.edge_mask, training=False) if node else self.edge_mask.sigmoid()
        node_mask = torch.zeros(node_feat_mask.shape[0])
        if node:
            node_feat_msg = torch.sum(node_feat_mask * data_sub.x, dim=-1).view(-1)
            x = data_sub.x.clone()
            x[x > 0.] = 1.
            node_feat_mask = node_feat_mask * x
            for n in range(N):
                if self.out_degree[n] > 0 or self.in_degree[n] > 0:
                    out_masks = torch.zeros(1)
                    if self.out_degree[n] > 0:
                        out_masks = edge_mask[self.out_edge_mask[n]]
                    node_mask_out = out_masks * node_feat_msg[n]
                    node_mask_out = self.agg1(node_mask_out)
                    in_masks = edge_mask[self.self_loop_mask[n]] if self.self_loop_mask is not None else torch.zeros(1)
                    node_mask_in = in_masks * node_feat_msg[n]
                    if self.in_degree[n] > 0:
                        in_nodes = data_sub.edge_index[0, self.in_edge_mask[n]]
                        in_masks = edge_mask[self.in_edge_mask[n]]
                        if self.self_loop_mask is not None:
                            in_masks = torch.cat((in_masks.view(-1), edge_mask[self.self_loop_mask[n]].view(-1)))
                        node_mask_in = in_masks * node_feat_msg[in_nodes]
                    node_mask_in = self.agg1(node_mask_in)
                    node_mask[n] = self.agg2(torch.cat((node_mask_in.view(-1), node_mask_out.view(-1))))
        else:
            node_mask = torch.zeros(N)
            for n in range(N):
                out_max = torch.tensor(0, dtype=torch.float)
                in_max = torch.tensor(0, dtype=torch.float)
                if self.out_degree[n] > 0:
                    out_max = torch.max(edge_mask[self.out_edge_mask[n]])
                if self.in_degree[n] > 0:
                    in_max = torch.max(edge_mask[self.in_edge_mask[n]])
                node_mask[n] = torch.max(out_max, in_max)
        self.__clear_masks__()

        # Scale node scores before returning.
        node_mask = self.scores_scaling_transform(node_mask)
        self.__clear_masks__()
        return node_feat_mask, edge_mask, node_mask, (node_idx, node_map)
    
    def integrated_gradients(self, data, target_idx, baseline=None, steps=50):
        """
        Compute integrated gradients for the node features for a given target index.
        """
        if baseline is None:
            baseline = torch.zeros_like(data.x)
        scaled_inputs = [baseline + float(i) / steps * (data.x - baseline) for i in range(steps + 1)]
        grads = []
        for x in scaled_inputs:
            x.requires_grad = True
            out = self.model(x)
            target = out[target_idx]
            self.model.zero_grad()
            target.backward(retain_graph=True)
            grads.append(x.grad.detach())
        avg_grads = torch.mean(torch.stack(grads), dim=0)
        integrated_grads = (data.x - baseline) * avg_grads
        
        return integrated_grads
    
    def explain_multi_scale(self, data, idx, loss_fc):
        """
        Generate a graph-level explanation by quantifying individual node contributions 
        to the overall model prediction. Instead of providing an explicit node mask from the
        explain_node method, this version focuses solely on integrated gradients to compute 
        node importance, and then derives an edge importance signal from these values.
        """
        # Compute integrated gradients for node features to capture each node's contribution.
        ig = self.integrated_gradients(data, idx)
        # Aggregate the integrated gradients for each node (e.g., mean across feature dimensions)
        node_importance = ig.abs().mean(dim=1)
        
        # Optionally, compute edge importance based on node contributions. 
        # For instance, an edge's importance could be the average importance of the two nodes it connects.
        edge_mask = self.compute_edge_importance(data, node_importance)
        
        # Additional information (could be useful for debugging or further analysis)
        sub_info = {"node_importance_raw": node_importance}
        
        # Since we are focusing on the graph-level explanation via node contributions,
        # we return None for any node-specific mask that we are no longer using.
        
        return None, edge_mask, node_importance, sub_info

    def counterfactual_analysis(self, data, idx, perturbation=0.1):
        """
        Perturb node features slightly and compare the prediction to evaluate
        the robustness of the explanation.
        """
        data_perturbed = data.clone()
        noise = torch.randn_like(data.x) * perturbation
        data_perturbed.x = data.x + noise
        out_orig = self.model(data.x)[idx]
        out_perturbed = self.model(data_perturbed.x)[idx]
        diff = torch.abs(out_orig - out_perturbed)

        return diff
    
    def scores_scaling_transform(self, scores, threshold=1e-6, alpha=2.0):
        """
        Apply a piecewise transformation where:
        - Scores below a given threshold (e.g., zeros) are set to 0.
        - Scores above the threshold are scaled and then raised to a power alpha.
        
        Parameters:
            scores (torch.Tensor): the input scores.
            threshold (float): values less than this are considered zeros.
            alpha (float): power factor applied to the scaled nonzero scores.
            
        Returns:
            torch.Tensor: transformed scores.
        """
        # Identify low values as "zero"
        mask = scores > threshold
        nonzero_values = scores[mask]
        
        if nonzero_values.numel() > 0:
            # Scale the nonzero values using min-max scaling (on nonzero subset)
            s_min = torch.min(nonzero_values)
            s_max = torch.max(nonzero_values)
            nonzero_scaled = (nonzero_values - s_min) / (s_max - s_min + 1e-12)
            # Apply power transform
            nonzero_transformed = nonzero_scaled ** alpha
            
            # Create new scores, keeping low values at zero
            new_scores = torch.zeros_like(scores)
            new_scores[mask] = nonzero_transformed
        else:
            new_scores = torch.zeros_like(scores)
        
        return new_scores

    def build_subgraph_from_nodes(self, data, node_indices):
        """
        Constructs a subgraph given the indices of nodes to include.
        """
        node_indices = torch.tensor(node_indices, dtype=torch.long)
        new_x = data.x[node_indices]
        new_edge_index, _ = subgraph(node_indices, data.edge_index.cpu(), relabel_nodes=True)
        
        return Data(x=new_x, edge_index=new_edge_index)

    def mask_nodes_in_data(self, data, selected_nodes):
        """
        Returns a new Data object with the same number of nodes as data.x,
        but with node features zeroed out for nodes that are not in selected_nodes.
        """
        new_data = data.clone()
        num_nodes = data.x.size(0)
        
        # Create an indicator mask with shape [num_nodes, 1].
        mask = torch.zeros(num_nodes, 1, device=data.x.device)
        mask[selected_nodes] = 1.0
        
        # Element-wise multiply to zero-out features of non-selected nodes.
        new_data.x = data.x * mask
        return new_data

    def minimal_subgraph_by_adding(self, data, node_importance, model, threshold=0.5):
        """
        Greedily adds nodes (starting with the highest scoring) until the subgraph
        produces a prediction equal to the full graph’s target prediction.
        In the returned submask (of shape [num_nodes]), the selected nodes are assigned 
        their respective importance values (and zero elsewhere). In any case at least one node is returned.
        
        Parameters:
            data: The original graph Data object.
            node_importance: A tensor with importance scores for each node.
            model: The trained GNN model.
            threshold: Threshold for binary predictions.
        
        Returns:
            A tuple (submask, selected_nodes, confidence_score, ep) where:
                - submask is a tensor of shape [num_nodes] with the node importance values at
                the selected indices and zeros elsewhere.
                - selected_nodes is the list of indices in the candidate subgraph.
                - confidence_score is the final raw model output on the masked graph.
                - ep is 1 if the candidate’s prediction matches the original target and 0 otherwise.
        """
        data = data.to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(data)
            if output.dim() > 1:
                target_label = output.argmax(dim=1)[0].item()
            else:
                target_label = (output > threshold).long().item()
        
        sorted_indices = torch.argsort(node_importance, descending=True).tolist()
        selected_nodes = []
        confidence_score = None
        ep = 0  # essentialness: 1 if candidate prediction matches target, 0 otherwise.
        
        for idx in sorted_indices:
            if idx in selected_nodes:
                continue
            selected_nodes.append(idx)
            masked_data = self.mask_nodes_in_data(data, selected_nodes)
            with torch.no_grad():
                confidence_score = model(masked_data).cpu()
                if confidence_score.dim() > 1:
                    pred = confidence_score.argmax(dim=1)[0].item()
                else:
                    pred = (confidence_score > threshold).long().item()
            if pred == target_label:
                ep = 1
                break
        
        # Ensure at least one node is selected.
        if len(selected_nodes) == 0:
            selected_nodes = [sorted_indices[0]]
            masked_data = self.mask_nodes_in_data(data, selected_nodes)
            with torch.no_grad():
                confidence_score = model(masked_data).cpu()
            if confidence_score.dim() > 1:
                pred = confidence_score.argmax(dim=1)[0].item()
            else:
                pred = (confidence_score > threshold).long().item()
            ep = 1 if pred == target_label else 0

        submask = torch.zeros(data.x.size(0))
        submask[selected_nodes] = node_importance[selected_nodes]
        
        return submask, selected_nodes, confidence_score, ep

    def minimal_subgraph_by_removal(self, data, node_importance, model, threshold=0.5):
        """
        Greedily removes nodes (starting with the least important) from the full graph
        while preserving the full-graph prediction. The output submask has nonzero values
        (the node importance) only for the nodes retained. A non-empty candidate is guaranteed.
        
        Parameters:
            data: The original graph Data object.
            node_importance: A tensor with importance scores for each node.
            model: The trained GNN model.
            threshold: Threshold for binary predictions.
        
        Returns:
            A tuple (submask, remaining_nodes, confidence_score, ep) where:
                - submask is a tensor of shape [num_nodes] with the node importance values at
                the remaining indices and zeros elsewhere.
                - remaining_nodes is the list of node indices that remain active.
                - confidence_score is the final raw model output on the masked graph.
                - ep is 1 if the candidate’s prediction equals the target and 0 otherwise.
        """
        data = data.to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(data)
            if output.dim() > 1:
                target_label = output.argmax(dim=1)[0].item()
            else:
                target_label = (output > threshold).long().item()
        
        all_nodes = list(range(data.x.size(0)))
        sorted_indices = torch.argsort(node_importance, descending=False).tolist()  # least important first
        remaining_nodes = all_nodes.copy()
        confidence_score = None
        
        for idx in sorted_indices:
            if idx not in remaining_nodes:
                continue
            candidate_nodes = remaining_nodes.copy()
            candidate_nodes.remove(idx)
            masked_data = self.mask_nodes_in_data(data, candidate_nodes)
            with torch.no_grad():
                confidence_score = model(masked_data).cpu()
                if confidence_score.dim() > 1:
                    pred = confidence_score.argmax(dim=1)[0].item()
                else:
                    pred = (confidence_score > threshold).long().item()
            if pred == target_label:
                remaining_nodes = candidate_nodes
            else:
                break
        
        # Ensure at least one node remains.
        if len(remaining_nodes) == 0:
            remaining_nodes = [torch.argsort(node_importance, descending=True)[0].item()]
            masked_data = self.mask_nodes_in_data(data, remaining_nodes)
            with torch.no_grad():
                confidence_score = model(masked_data).cpu()
            if confidence_score.dim() > 1:
                pred = confidence_score.argmax(dim=1)[0].item()
            else:
                pred = (confidence_score > threshold).long().item()
        
        submask = torch.zeros(data.x.size(0))
        submask[remaining_nodes] = node_importance[remaining_nodes]
        
        # Compute final prediction for EP.
        with torch.no_grad():
            final_masked = self.mask_nodes_in_data(data, remaining_nodes)
            final_conf = model(final_masked).cpu()
            if final_conf.dim() > 1:
                final_pred = final_conf.argmax(dim=1)[0].item()
            else:
                final_pred = (final_conf > threshold).long().item()
        ep = 1 if final_pred == target_label else 0
        
        return submask, remaining_nodes, confidence_score, ep

    def optimal_minimal_subgraph(self, data, node_importance, model, threshold=0.5):
        """
        Iteratively adds nodes (using the descending order of importance) and records each candidate
        subgraph that produces the target prediction. After all candidates are evaluated, the function
        returns the candidate with the best optimization criterion:
        
        - If the target label is 1 (vulnerable): choose the candidate with the highest confidence.
        - If the target label is 0 (benign): choose the candidate with the lowest confidence.
        
        If no candidate meeting the target prediction is found, the function returns at least the candidate
        with the highest importance (i.e. a single-node subgraph). A non-empty output is always guaranteed.
        
        Parameters:
            data: The original graph Data object.
            node_importance: A tensor with importance scores for each node.
            model: The trained GNN model.
            threshold: Threshold for binary predictions.
        
        Returns:
            A tuple (submask, candidate_nodes, best_confidence_score, ep) where:
                - submask is a tensor of shape [num_nodes] with nonzero values (the node importance)
                for nodes in the optimal candidate.
                - candidate_nodes is the list of node indices for the optimal candidate.
                - best_confidence_score is the raw output (as a float) for that candidate.
                - ep is 1 if the candidate's prediction equals the original target, 0 otherwise.
        """
        data = data.to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(data)
            if output.dim() > 1:
                target_label = output.argmax(dim=1)[0].item()
            else:
                target_label = (output > threshold).long().item()
        
        sorted_indices = torch.argsort(node_importance, descending=True).tolist()
        candidate_nodes = []
        candidates = []       # list of (candidate_nodes, confidence_score) for those that meet target.
        
        # Iterate over candidates by adding nodes one by one.
        for idx in sorted_indices:
            candidate_nodes.append(idx)
            masked_data = self.mask_nodes_in_data(data, candidate_nodes)
            with torch.no_grad():
                confidence_score = model(masked_data).cpu()
                if confidence_score.dim() > 1:
                    pred = confidence_score.argmax(dim=1)[0].item()
                else:
                    pred = (confidence_score > threshold).long().item()
            if pred == target_label:
                # Save candidate (deep copy the candidate_nodes list).
                candidates.append((candidate_nodes.copy(), confidence_score.item() if (confidence_score.dim() == 0 or confidence_score.numel() == 1)
                                else confidence_score[0].item()))
        
        # If no candidate produced the target prediction, default to the best single node.
        if not candidates:
            best_candidate = [sorted_indices[0]]
            masked_data = self.mask_nodes_in_data(data, best_candidate)
            with torch.no_grad():
                confidence_score = model(masked_data).cpu()
            if (confidence_score.dim() == 0 or confidence_score.numel() == 1):
                best_score = confidence_score.item()
            else:
                best_score = confidence_score[0].item()
            ep = 1 if ((confidence_score.argmax(dim=1)[0].item() if confidence_score.dim() > 1
                        else (confidence_score > threshold).long().item()) == target_label) else 0
        else:
            # Choose the candidate with optimal confidence.
            # For target 1 (vulnerable), choose highest confidence; for target 0 (benign), choose lowest.
            if target_label == 1:
                best_candidate, best_score = max(candidates, key=lambda x: x[1])
            else:
                best_candidate, best_score = min(candidates, key=lambda x: x[1])
            ep = 1  # Because candidate's prediction equals target.
        
        # Build submask from the best candidate.
        submask = torch.zeros(data.x.size(0))
        if len(best_candidate) == 0:
            best_candidate = [sorted_indices[0]]
        submask[best_candidate] = node_importance[best_candidate]
        
        return submask, best_candidate, best_score, ep
    
    def branch_pruned_subgraph(self, data, node_importance, model, threshold=0.5):
        """
        Using branch-based pruning, iteratively removes nodes from an initial candidate subgraph
        (obtained via a removal strategy) by checking branch leaves from a spanning tree. The idea
        is to remove nodes from the outside in (i.e. pruning the branches) rather than simply removing
        the least important node. The resulting candidate is guaranteed to be non-empty.

        Parameters:
            data: The original graph Data object.
            node_importance: A tensor with importance scores for each node.
            model: The trained GNN model.
            threshold: Threshold for binary predictions.

        Returns:
            A tuple (submask, pruned_nodes, confidence_score, ep) where:
            - submask is a tensor of shape [num_nodes] with the node importance values at the selected indices and 0 elsewhere.
            - pruned_nodes is the list of node indices in the final candidate subgraph.
            - confidence_score is the final raw model output on the masked graph.
            - ep is 1 if the candidate’s prediction equals the full graph prediction (i.e. EP = 1), else 0.
        """
        # Send data to device and evaluate original prediction.
        data = data.to(self.device)
        model.eval()
        with torch.no_grad():
            full_output = model(data)
            if full_output.dim() > 1:
                target_label = full_output.argmax(dim=1)[0].item()
            else:
                target_label = (full_output > threshold).long().item()
        
        # First, get an initial candidate using a removal-based strategy.
        # (This is similar to your greedy minimal_subgraph_by_removal.)
        all_nodes = list(range(data.x.size(0)))
        sorted_indices = torch.argsort(node_importance, descending=False).tolist()  # least important first
        candidate_nodes = all_nodes.copy()
        confidence_score = None

        for idx in sorted_indices:
            if idx not in candidate_nodes:
                continue
            temp_candidate = candidate_nodes.copy()
            temp_candidate.remove(idx)
            masked_data = self.mask_nodes_in_data(data, temp_candidate)
            with torch.no_grad():
                temp_confidence = model(masked_data).cpu()
                if temp_confidence.dim() > 1:
                    pred = temp_confidence.argmax(dim=1)[0].item()
                else:
                    pred = (temp_confidence > threshold).long().item()
            if pred == target_label:
                candidate_nodes = temp_candidate  # removal accepted

        # Ensure at least one node remains.
        if len(candidate_nodes) == 0:
            candidate_nodes = [max(all_nodes, key=lambda i: node_importance[i].item())]

        # Now perform branch pruning on the initial candidate.
        pruned_nodes = candidate_nodes.copy()
        improved = True
        while improved:
            improved = False
            # Build masked data from current candidate.
            candidate_data = self.mask_nodes_in_data(data, pruned_nodes)
            
            # Build NetworkX graph from candidate_data.
            # Here we assume that data.edge_index is a tensor of shape [2, num_edges].
            candidate_set = set(pruned_nodes)
            edge_index = candidate_data.edge_index.cpu().numpy()
            # Filter edges that connect only nodes in candidate_set.
            filtered_edges = [(u, v) for u, v in zip(edge_index[0], edge_index[1]) 
                            if u in candidate_set and v in candidate_set]
            if len(filtered_edges) == 0 or len(pruned_nodes) <= 1:
                break  # no further structure to prune.
            G = nx.Graph()
            G.add_nodes_from(pruned_nodes)
            G.add_edges_from(filtered_edges)
            
            # Compute a spanning tree. (You may also consider a breadth-first tree.)
            T = nx.minimum_spanning_tree(G)
            # Identify leaves (degree <= 1).
            leaves = [node for node in T.nodes() if T.degree(node) <= 1]
            
            for leaf in leaves:
                if len(pruned_nodes) <= 1:
                    continue  # do not remove if only one node remains.
                new_candidate = pruned_nodes.copy()
                new_candidate.remove(leaf)
                new_masked_data = self.mask_nodes_in_data(data, new_candidate)
                with torch.no_grad():
                    new_confidence = model(new_masked_data).cpu()
                    if new_confidence.dim() > 1:
                        new_pred = new_confidence.argmax(dim=1)[0].item()
                    else:
                        new_pred = (new_confidence > threshold).long().item()
                if new_pred == target_label:
                    pruned_nodes = new_candidate
                    improved = True
                    # Once one leaf is removed, restart the branch-pruning loop.
                    break
        
        # Final candidate after branch pruning.
        final_candidate = pruned_nodes
        final_masked_data = self.mask_nodes_in_data(data, final_candidate)
        with torch.no_grad():
            final_confidence = model(final_masked_data).cpu()
            if final_confidence.dim() > 1:
                final_pred = final_confidence.argmax(dim=1)[0].item()
            else:
                final_pred = (final_confidence > threshold).long().item()
        ep = 1 if final_pred == target_label else 0
        confidence_score = final_confidence.item() if (final_confidence.dim() == 0 or final_confidence.numel() == 1) else final_confidence[0].item()
        
        # Build the submask: assign each selected node its corresponding importance.
        submask = torch.zeros(data.x.size(0))
        if len(final_candidate) == 0:
            final_candidate = [max(all_nodes, key=lambda i: node_importance[i].item())]
        submask[final_candidate] = node_importance[final_candidate]
        
        return submask, final_candidate, confidence_score, ep

    def counterfactual_graph_explanation(self, data, node_importance, model, threshold=0.5):
        """
        Generates a graph-level counterfactual explanation by removing branches from the 
        full graph. Instead of removing one node at a time based on node-level importance,
        this method computes a spanning tree of the candidate subgraph and then removes one or
        more leaves (i.e. an entire branch) in each iteration. The process stops once the model's
        prediction flips relative to the original prediction.
        
        Parameters:
            data: The original graph Data object.
            model: The trained GNN model.
            threshold: Threshold for binary decisions (if applicable).
        
        Returns:
            A tuple (cf_submask, removed_nodes, cf_confidence, cf_prediction) where:
            - cf_submask is a tensor of shape [num_nodes] with ones for nodes kept in the candidate 
                counterfactual subgraph and zeros for nodes removed.
            - removed_nodes is the list of node indices that have been removed (i.e. pruned from the graph).
            - cf_confidence is the final raw model output (confidence score) on the candidate subgraph.
            - cf_prediction is the predicted label for the candidate subgraph.
        """
        # Send data to device and get original prediction.
        data = data.to(self.device)
        model.eval()
        with torch.no_grad():
            full_output = model(data)
            if full_output.dim() > 1:
                orig_pred = full_output.argmax(dim=1)[0].item()
            else:
                orig_pred = (full_output > threshold).long().item()

        # Start with all nodes active.
        candidate_nodes = list(range(data.x.size(0)))
        removed_nodes = []  # to store nodes that are pruned.
        
        # We want to iterate until the prediction flips or we cannot prune any further.
        candidate_changed = True
        while candidate_changed and len(candidate_nodes) > 1:
            candidate_changed = False
            # Build candidate subgraph.
            candidate_data = self.mask_nodes_in_data(data, candidate_nodes)
            
            # Construct a NetworkX graph from candidate_data.
            candidate_set = set(candidate_nodes)
            edge_index = candidate_data.edge_index.cpu().numpy()  # shape [2, num_edges]
            filtered_edges = [(u, v) for u, v in zip(edge_index[0], edge_index[1]) 
                            if u in candidate_set and v in candidate_set]
            if len(filtered_edges) == 0 or len(candidate_nodes) <= 1:
                break  # No structure remains to prune.
            G = nx.Graph()
            G.add_nodes_from(candidate_nodes)
            G.add_edges_from(filtered_edges)
            # Compute a spanning tree, which gives a sense of the “backbone” structure.
            T = nx.minimum_spanning_tree(G)
            # Identify leaves in the spanning tree (nodes with degree <= 1).
            leaves = [node for node in T.nodes() if T.degree(node) <= 1]
            # Track the best candidate update among the leaves if no immediate flip is found.
            best_temp_candidate = None
            best_temp_confidence = None
            candidate_updated = False
            
            for leaf in leaves:
                # Try removing this leaf.
                temp_candidate = candidate_nodes.copy()
                temp_candidate.remove(leaf)
                temp_data = self.mask_nodes_in_data(data, temp_candidate)
                with torch.no_grad():
                    temp_output = model(temp_data).cpu()
                    if temp_output.dim() > 1:
                        temp_pred = temp_output.argmax(dim=1)[0].item()
                    else:
                        temp_pred = (temp_output > threshold).long().item()
                # Check if the prediction flips.
                if temp_pred != orig_pred:
                    # Immediately accept this candidate.
                    candidate_nodes = temp_candidate
                    removed_nodes.append(leaf)
                    candidate_changed = True
                    candidate_updated = True
                    break  # Stop iterating over leaves.
                else:
                    # Otherwise, record the candidate that brings the confidence closer to flip.
                    # For example, if orig_pred == 1 (vulnerable), we want the confidence to drop.
                    # (We assume here that a lower confidence value moves toward the benign class.)
                    conf_value = temp_output.item() if (temp_output.dim() == 0 or temp_output.numel() == 1) else temp_output[0].item()
                    if best_temp_confidence is None:
                        best_temp_confidence = conf_value
                        best_temp_candidate = temp_candidate.copy()
                        temp_leaf = leaf
                    else:
                        if orig_pred == 1 and conf_value < best_temp_confidence:
                            best_temp_confidence = conf_value
                            best_temp_candidate = temp_candidate.copy()
                            temp_leaf = leaf
                        elif orig_pred == 0 and conf_value > best_temp_confidence:
                            best_temp_confidence = conf_value
                            best_temp_candidate = temp_candidate.copy()
                            temp_leaf = leaf
            # If no immediate flip was found but one candidate improved the direction of change, update candidate_nodes.
            if not candidate_updated and best_temp_candidate is not None:
                candidate_nodes = best_temp_candidate
                removed_nodes.append(temp_leaf)
                candidate_changed = True
            
            # Check if the updated candidate now flips the prediction.
            candidate_data = self.mask_nodes_in_data(data, candidate_nodes)
            with torch.no_grad():
                current_output = model(candidate_data).cpu()
                if current_output.dim() > 1:
                    current_pred = current_output.argmax(dim=1)[0].item()
                else:
                    current_pred = (current_output > threshold).long().item()
            if current_pred != orig_pred:
                # Once prediction is flipped, we have a counterfactual.
                break
        
        # Ensure candidate is not empty.
        if len(candidate_nodes) == 0:
            # Fallback: keep the single node with highest degree in the full graph.
            candidate_nodes = [max(range(data.x.size(0)), key=lambda i: 1)]
        
        # Build the counterfactual submask.
        cf_submask = torch.zeros(data.x.size(0))
        cf_submask[candidate_nodes] = node_importance[candidate_nodes]
        # Get final confidence and prediction.
        with torch.no_grad():
            final_data = self.mask_nodes_in_data(data, candidate_nodes)
            final_output = model(final_data).cpu()
            if final_output.dim() > 1:
                cf_prediction = final_output.argmax(dim=1)[0].item()
            else:
                cf_prediction = (final_output > threshold).long().item()
        cf_confidence = (final_output.item() if final_output.dim() == 0 or final_output.numel() == 1 
                        else final_output[0].item())

        
        return cf_submask, removed_nodes, cf_confidence, cf_prediction
