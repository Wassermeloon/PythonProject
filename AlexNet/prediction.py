
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

def main():

    # pre processing

    # hyper parameters
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    data_transform = transforms.Compose([transforms.Resize(254),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)]
                                        )

    # loade image

    root = "E:\\code\\PythonProject\\daisy.jpg"
    img = Image.open(root)
    plt.imshow(img)

    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    try:
        json_file = open('./class_indice.json')
        class_indict = json.load((json_file))

    except Exception as e:
        print(e)
        exit(-1)


    model = AlexNet(num_classes=5)

    model_weight_path = "./AlexNet.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()

    with torch.no_grad():
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print("The predicted class is {} || the accuracy is {:.4f} %".format(class_indict[str(predict_cla)], predict[predict_cla].item()))

    plt.show()

    # prediction by tensorflow framework

    # from model_tensorflow import Alex_v1
    # from PIL import Image
    # import numpy as np
    # import json
    # import matplotlib.pyplot as plt
    #
    # im_height = 224
    # im_width = 224
    #
    # # load Image
    # sample_root = "E:\\code\\PythonProject\\daisy.jpg"
    # img = Image.open(sample_root)
    # print(type(img))
    # # resize image into 224x224
    # img = img.resize((im_width, im_height))
    # plt.imshow(img)
    #
    # # scaling pixel value to (0-1）
    #
    # img = np.array(img) / 255.0
    #
    # # add a batch channel
    # img = (np.expand_dims(img, axis=0))
    # try:
    #     json_file = open("./class_indices.json", "r")
    #     class_indices = json.load(json_file)
    #
    # except Exception as e:
    #     print(e)
    #     exit(-1)
    #
    # model = Alex_v1(class_num=5)
    # model.load_weights("./save_weights/AlexNet.ckpt")
    # outputs = model(img)
    # outputs = np.squeeze(outputs)
    # pred = np.argmax(outputs)
    #
    # print("The predicted class is {} || the accuracy is {:.4f} %".format(class_indices[str(pred)], outputs[pred]*100))
    #
    # plt.show()

if __name__ == '__main__':
    main()