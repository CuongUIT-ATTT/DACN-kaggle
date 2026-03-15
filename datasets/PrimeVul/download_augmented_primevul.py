import requests
import zipfile
import os

# Replace with your actual OneDrive shareable link
onedrive_url = "https://upcomillas-my.sharepoint.com/personal/201805328_alu_comillas_edu/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2F201805328%5Falu%5Fcomillas%5Fedu%2FDocuments%2FMaryland%2FRESEARCH%2FDataset%2Fcwe20cfa%5FCWE%2D20%5Faugmented%2Ezip"

def get_direct_download_url(share_url):
    """Follows the redirect to get the actual download URL."""
    response = requests.head(share_url, allow_redirects=True)
    return response.url

def download_zip(share_url, output_path):
    """Downloads the zip file from the resolved URL."""
    direct_url = get_direct_download_url(share_url)
    print(f"Downloading from: {direct_url}")
    response = requests.get(direct_url, stream=True)
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to: {output_path}")

def unzip_file(zip_path, extract_to='.'):
    """Unzips the downloaded file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to: {os.path.abspath(extract_to)}")

# File paths
zip_filename = "augmented_cwe20cfa.zip"

# Run the process
download_zip(onedrive_url, zip_filename)
unzip_file(zip_filename)