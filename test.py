import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import io

from extract_cnn_vgg16_keras import VGGNet
img = []
ques = []
final_answer = []
final_image = []
input = io.open("./test_where_2018.txt","r",encoding="UTF-8")
for l in input.readlines():
    line = l.split("|")
    image,question = line[0],line[1]
    question = question.strip("\n")
    img.append(image)
    ques.append(question)
for glove_i in range(len(img)):
    query = './test/VQAMed2018Test-images/JPN-5-42-g001.jpg'
    1471-2253-12-8-1
    index = 'models/densenet_featureCNN.h5'
    result = 'Train_where_images'
# read in indexed images' feature vectors and corresponding image names
    h5f = h5py.File(index, 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()


    print("--------------------------------------------------")
    print("               searching starts")
    print("--------------------------------------------------")

# read and show query image
    queryImg = mpimg.imread(query)
    plt.title("Query Image")
    print("no.",glove_i)
    plt.imshow(queryImg)
    plt.show()

# init VGGNet16 model
    model = VGGNet()

# extract query image's feature, compute simlarity score and sort
    queryVec = model.densenet_extract_feat(query)
    scores = np.dot(queryVec, feats.T)
#    scores = np.dot(queryVec, feats.T)/(np.linalg.norm(queryVec)*np.linalg.norm(feats.T))
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
# print (rank_ID)
    print(rank_score)

# number of top retrieved images to show
    maxres = 5
    imlist = []
    imques = []
    imansw = []
    imques1 = []
    imansw1 = []
    im = []


    for i, index in enumerate(rank_ID[0:maxres]):
        imlist.append(str(imgNames[index],encoding=('UTF-8')))
#    print(type(str(imgNames[index],encoding=('UTF-8'))))
#    raise ValueError
        print("image names: " + str(imgNames[index]) + " scores: %f" % rank_score[i])
    print("top %d images in order are: " % maxres, imlist)
    input = io.open("E:/search-picture-2018/data/train/VQAMed2018Train-QA-where.txt","r",encoding="UTF-8")
    for l in input.readlines():
        line = l.split("|")
        image,question,answer = line[0],line[1],line[2]
        im.append(line[0]+".jpg")
        imques.append(line[1])
        imansw.append(line[2])
    input.close()
    index = []
    for i in range(len(imlist)):
        a = im.index(imlist[i])
        index.append(a)
#    print(im[a])
##print(index)
#raise ValueError
    imques2 = []
    imansw2 = []
    for j in range(len(index)):
        imques2.append(imques[index[j]])
        imansw2.append(imansw[index[j]])

    final_answer.append(imansw2[0])
    final_image.append(imlist[0])
print("final_answer:",final_answer)
print("finish.........")  
print(len(final_answer))
output =io.open("./test/test_densenet_where.txt","w",encoding="UTF-8")
output1 =io.open("./test/test_densenet_where.txt","w",encoding="UTF-8")
for line in final_answer:
    output.write(line)
output.close()
for line in final_image:
    output1.write(line+"\n")
output1.close()
print("down....")
id = 0
index = 0
input = io.open("./test/test_densenet_where.txt","r",encoding="UTF-8")
output = io.open("./test/test_densenet_where_part.txt","w",encoding="UTF-8")
for l in input.readlines():
    id += 1
    line = l.split("\n")
    image_id = img[index]
    index += 1
    line = line[0]
    output.write(str(id)+"\t"+image_id+"\t"+line+"\n")
output.close()
print("succeed.....")