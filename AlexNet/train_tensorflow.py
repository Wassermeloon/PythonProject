from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model_tensorflow import Alex_v1
import tensorflow as tf
import json
import os


def main():
    data_list = ["train", "val"]
    data_root = os.path.abspath(os.path.join(os.getcwd(),".."))
    image_path = os.path.join(data_root, "data_set", "flower_data")
    data_dir = {x: os.path.join(image_path, x) for x in data_list}
    print(data_dir['train'])

    # create direction for saving weights

    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10
    learning_rate = 0.0005

    # use ImageDataGenerator to create a DataLoader
    img_generator = {x: ImageDataGenerator(rescale=1./225, horizontal_flip=True)
                     for x in data_list} # create workspace and have some predefined operation
    data_gen = {x: img_generator[x].flow_from_directory(directory=data_dir[x],
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        target_size=(im_height, im_width),
                                                        class_mode='categorical') for x in data_list} # read datafile


    # get the total amount of data
    data_len = {x: data_gen[x].n for x in data_list}

    # get class dic
    class_indices = data_gen["train"].class_indices
    class_indices = dict((value, key) for key, value in class_indices.items()) # for later easier use

    # write class into json file
    json_str = json.dumps(class_indices, indent=4)
    with open("class_indice.json", 'w') as json_file:
        json_file.write(json_str)

    # show image
    sample_training_images, sample_training_labels = next(data_gen["train"])

    def plotImage(images):
        f, ax = plt.subplots(2, 2, figsize=(20, 20))
        ax = ax.flatten()
        for img, ax in zip(images, ax):
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    plotImage(sample_training_images[:4])

    model = Alex_v1(im_height=im_height, im_width=im_width,class_num=5)
    model.summary() # to see the parameters of model

    # # training process
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # model.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])
    # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlexTensorflow.h5',
    #                                                 save_best_only=True,
    #                                                 save_weights_only=True,
    #                                                 monitor='val_loss')]
    # history = model.fit(x=data_gen["train"],
    #                     steps_per_epoch=data_len["train"] // batch_size,
    #                     epochs=epochs,
    #                     validation_data=data_gen["val"],
    #                     validation_steps=data_len["val"] // batch_size,
    #                     callbacks=callbacks)

    # # plot loss and accuracy image
    # history_dict = history.history
    # train_loss = history_dict["loss"]
    # train_accuracy = history_dict["accuracy"]
    # val_loss = history_dict["val_loss"]
    # val_accuracy = history_dict["val_accuracy"]

    # # figure
    # plt.figure()
    # plt.plot(range(epochs), train_loss, label="train_loss")
    # plt.plot(range(epochs), val_loss, label="val_loss")
    # plt.legend()
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    #
    # plt.figure()
    # plt.plot(range(epochs), train_accuracy, label="train_accuracy")
    # plt.plot(range(epochs), val_accuracy, label="val_accuracy")
    # plt.legend()
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.grid()
    # plt.show()

    # using keras low level api for training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss_value = loss_object(labels, predictions)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss_value)
        train_accuracy(labels, predictions)

    @tf.function
    def val_step(images, labels):
        predictions = model(images, training=False)
        loss_value = loss_object(labels, predictions)

        val_loss(loss_value)
        val_accuracy(labels, predictions)

    best_val_accuracy = 0.0
    test_loss_list = list()
    test_acc_list = list()
    val_loss_list = list()
    val_acc_list = list()

    # training process
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for x in data_list:
            # if x == "train":
            #     for train_images, train_labels in data_gen["train"]:
            #         train_step(train_images, train_labels)
            # else:
            #     for val_images, val_labels in data_gen["val"]:
            #         val_step(val_images, val_labels)

            if x == "train":
                for step in range(data_len["train"] // batch_size):
                    train_images, train_labels = next(data_gen["train"])
                    train_step(train_images, train_labels)
            else:
                for step in range(data_len["val"] // batch_size):
                    val_images, val_labels = next(data_gen["val"])
                    val_step(val_images, val_labels)

            if val_accuracy.result() >  best_val_accuracy:
                best_val_accuracy = val_accuracy.result()
                model.save_weights("./save_weights/AlexNet.ckpt", save_format='tf')

        # data for figure
        test_loss_list.append(train_loss.result())
        test_acc_list.append(train_accuracy.result())
        val_loss_list.append(val_loss.result())
        val_acc_list.append(val_accuracy.result())

        template = "Epoch {} || Training process: loss{:.4f}, accuracy: {:.4f} % || " \
                   "Validation process: loss {:.4f}, validation accuracy {:.4f} %"
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100,
                              val_loss.result(), val_accuracy.result()*100))
        print("--"*35)


    # plot loss and accuracy image
    plt.figure()
    plt.plot(range(len(test_loss_list)), test_loss_list, label="train_loss")
    plt.plot(range(len(val_loss_list)), val_loss_list, label="val_loss")
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Loss figure")
    plt.grid()

    plt.figure()
    plt.plot(range(len(test_acc_list)), test_acc_list, label="train_accuracy")
    plt.plot(range(len(val_acc_list)), val_acc_list, label="val_accuracy")
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title("Accuracy figure")
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()