import os

train = open('barricade/ImageSets/train.txt', 'w')
test = open('barricade/ImageSets/test.txt', 'w')
label_list = open('barricade/label_list.txt', 'w')
xml_path = "./Annotations/"
img_path = "./JPEGImages/"
count = 0

for xml_name in os.listdir('barricade/Annotations'):
    data =img_path + xml_name[:-4] + ".jpg " + xml_path + xml_name + "\n"
    if(count%10==0):
        test.write(data)
    else:
        train.write(data)
    count += 1

label_list.write("barricade")

train.close()
test.close()
label_list.close()