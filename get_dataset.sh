# Download dataset from Dropbox
wget --no-check-certificate 'https://www.dropbox.com/s/0eywff6askh1wq3/Medical_data.zip?dl=0' -O 'Medical_data.zip'
wget --no-check-certificate 'https://www.dropbox.com/s/83yo5tivkojvcom/input_video.zip?dl=0' -O 'input_video.zip'

# Unzip the downloaded zip file
unzip ./Medical_data.zip
unzip ./input_video.zip

# Remove the downloaded zip file
rm ./Medical_data.zip
rm ./input_video.zip