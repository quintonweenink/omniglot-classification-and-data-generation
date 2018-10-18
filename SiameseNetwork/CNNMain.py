import time
import CNN
# import Siamese
import torch
import CNNDataPreparation as dp
import torch.nn.functional as F


def train_cnn(cnet, d, epochs, learn_rate):
    num_batches = len(d.train_loader)
    print("NUM_BATCHES:", num_batches)
    loss, optimiser = cnet.module.loss_and_optimiser(learn_rate)
    s_time = time.time()
    total_train_loss = 0
    itot = 0

    for epoch in range(epochs):
        cnet.train()
        running_loss = 0.0
        print_every = num_batches // 10
        correct = 0
        total_correct = 0
        start_time = time.time()
        s_t = time.time()
        counter = 0
        run_counter = 0
        for i, data in enumerate(d.train_loader, 0):
            inputs, labels = data
            # inputs[0] = inputs[0].cuda()
            # inputs[1] = inputs[1].cuda()
            # labels = labels.cuda()
            inputs, labels = inputs.cuda(), labels.cuda()
            # print(labels)
            optimiser.zero_grad()
            outputs = cnet(inputs)
            # print(outputs, torch.min(outputs), torch.max(outputs))
            loss_out = loss(outputs, labels)
            run_counter += d.batch_size
            counter += d.batch_size

            loss_out.backward()
            optimiser.step()
            running_loss += loss_out.item()
            total_train_loss += loss_out.item()
            # print(outputs, outputs.size())
            predicted = torch.Tensor([torch.argmax(outputs[v]) for v in range(0, len(outputs))]).long()
            corr = predicted.eq(labels.cpu()).sum().item()
            correct += corr

            total_correct += corr
            itot += 1
            if (i + 1) % print_every == 0:
                # print(print_every, itot)
                print("Epoch {}, {:d}% \t running_loss: {:.4f} running acc: {}/{} ({:.4f}%) \ttotal_loss: {:.4f} "
                      "total_acc: {}/{} ({:.4f}%) took {:.2f}s".format(epoch + 1, int(100 * (i + 1) / num_batches),
                                                                       running_loss / print_every, correct,
                                                                       run_counter, 100 * (correct / run_counter),
                                                                       total_train_loss / itot, total_correct,
                                                                       counter,
                                                                       100 * (total_correct / counter),
                                                                       (time.time() - start_time)))
                correct = 0
                running_loss = 0.0
                run_counter = 0
                start_time = time.time()
        correct = 0
        total_val_loss = 0
        cnet.eval()
        for inputs, labels in d.val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            val_outputs = cnet(inputs)
            # print(labels)
            val_loss = loss(val_outputs, labels)

            total_val_loss += val_loss.item()
            predicted = torch.Tensor([torch.argmax(val_outputs[v]) for v in range(0, len(val_outputs))]).long()
            # print(predicted)
            correct += predicted.eq(labels.cpu()).sum().item()

        tot = int(len(d.val_loader.dataset)*0.2)
        tota = len(d.val_loader)

        print("Validation loss = {:.6f} Accuracy = {}/{} ({:.6f}%)".format(total_val_loss / tota, correct, tot,
                                                                           100 * (correct / tot)))

        print("Epoch and Validation took {:.2f}s".format(time.time() - s_t))
    print("Training finished, took {:.2f}s".format(time.time() - s_time))


def test(cnet, d):
    cnet.eval()
    total = 0
    correct = 0
    t_loss = 0.0

    with torch.no_grad():
        for inputs, target in d.test_loader:
            inputs, target = inputs.cuda(), target.cuda()
            test_outputs = cnet(inputs)

            tt = F.cross_entropy(test_outputs, target, reduce=False)
            t_loss += torch.sum(tt)
            predicted = torch.Tensor([torch.argmax(test_outputs[v]) for v in range(0, len(test_outputs))]).long()
            correct += predicted.eq(target.cpu()).sum().item()

    total = len(d.test_loader.dataset)
    t_loss /= total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(t_loss, correct, total, 100. * (correct
                                                                                 / total)))


if __name__ == "__main__":
    args = {'out1': 16, 'out2': 32}  # , 'out3': 64, 'out4': 128}
    # for z in range(0, 1):
    #     learning_rate = 0.001
    #     if torch.cuda.device_count() > 0:
    #         print("USING {:1d} GPU('s)".format(torch.cuda.device_count()))
    #         dids = [i for i in range(0, torch.cuda.device_count())]
    #         print("Devices:", dids)
    #         net = torch.nn.DataParallel(CNN.CNN(**args), device_ids=dids).float().cuda()
    #     else:
    #         net = CNN.CNN(**args).float()
    #     d_prep = dp.DataPrep()
    #     train_cnn(net, d=d_prep, epochs=20, learn_rate=learning_rate)
    #     test(net, d_prep)
    for z in range(0, 1):
        learning_rate = 0.001
        if torch.cuda.device_count() > 0:
            print("USING {:1d} GPU('s)".format(torch.cuda.device_count()))
            dids = [i for i in range(0, torch.cuda.device_count())]
            print("Devices:", dids)
            net = torch.nn.DataParallel(CNN.CNN(**args), device_ids=dids).float().cuda()
        else:
            net = CNN.CNN(**args).float()

        d_prep = dp.DataPrep()
        train_cnn(net, d=d_prep, epochs=20, learn_rate=learning_rate)
        test(net, d_prep)
