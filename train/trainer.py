import torch
from tqdm import tqdm 
import os

def train(trainloader, testloader, model, optimizer, criterion, epochs, device, target_accuracy=None, model_save_path='./saved'):
    best_acc = 0.0
    train_accuracies = []  # record accuracy for training
    train_losses = []  # record loss for training
    valid_accuracies = []  # record accuracy for validation
    valid_losses = []  # record loss for validation
    folder_path = os.path.dirname(model_save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        running_correct = 0   #count number of correct prediction in this batch
        total_train = 0   #count of all data in this batch

        train_bar=tqdm(trainloader,position=0,leave=True) #file=sys.stdout
        for i, data in enumerate(train_bar):
            images, labels, mask, info = data # get the inputs; data is a list of [images, labels]
            optimizer.zero_grad() # zero the parameter gradients
            # forward + backward + optimize
            preds_scores = model(images.to(device)) #output=logit
            _, preds_class = torch.max(preds_scores, 1) #igoring the max of every row but return the max of every column, which is the class
            #preds_class = torch.argmax(preds_scores, dim=-1)
            loss = criterion(preds_scores, labels.to(device))
            loss.backward() # Backpropagation
            optimizer.step() # Update the weights

            running_loss += loss.item() #batch
            running_correct += (preds_class == labels.to(device)).sum().item() #torch.sum(preds_class == labels)
            total_train += labels.size(0)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs,
                                                                    loss)

        # print statistics
        train_accuracy = running_correct / total_train #epoch_acc, double for better accuracy
        train_accuracies.append(train_accuracy)
        train_loss = running_loss / len(trainloader) #epoch_loss
        train_losses.append(train_loss)
        

        if target_accuracy != None:
            if train_accuracy > target_accuracy:
                print("Early Stopping")
                break

        model.eval()
        correct_valid = 0
        total_valid = 0
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(testloader)
            for val_data in val_bar:
                val_images, val_labels, val_mask, val_info = val_data
                preds_scores = model(val_images.to(device)) #preds_probs,_ = model(inputs)
                # loss = loss_function(outputs, test_labels)
                _, preds_class = torch.max(preds_scores[0], 1)  
                # torch.max(preds_scores, dim=1)[1]
                correct_valid += torch.eq(preds_class, val_labels.to(device)).sum().item()
                total_valid += val_labels.size(0) #=len(testset)
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                            epochs)

        valid_accuracy = correct_valid / total_valid
        valid_accuracies.append(valid_accuracy)
        #valid_losses.append(loss.item())

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                (epoch + 1, train_loss, valid_accuracy)) #transteps=len(trainloader)
        # s per iteration, count of iteration = total number of sample / batch size

        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            torch.save(model.state_dict(), model_save_path)

    print('Finished Training')
    return train_losses, valid_accuracies