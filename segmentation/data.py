
# script to download an example set of prostate MR images anf their gland segmentation as labels
import os


DATA_PATH = './data'
RESULT_PATH = './result'

temp_file = os.path.join(DATA_PATH,'datasets-promise12.zip')
origin = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/archive/promise12/datasets-promise12.zip'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

os.system("wget -P %s  %s" % (DATA_PATH, origin))
os.system('unzip %s -d %s' % (temp_file, DATA_PATH))
os.remove(temp_file)

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

print('Promise12 data downloaded: <%s>.' % os.path.abspath(os.path.join(DATA_PATH,'datasets-promise12')))
print('Result directory created: <%s>.' % os.path.abspath(RESULT_PATH))
