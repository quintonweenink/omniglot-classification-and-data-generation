import time
# import CNN
import Siamese
import torch
import DataPreparation as dp
import torch.nn.functional as F

yesno = lambda val: 0 if val < 0.5 else 1


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
        correct = [0, 0]
        total_correct = [0, 0]
        start_time = time.time()
        s_t = time.time()
        counter = [0, 0]
        run_counter = [0, 0]
        for i, data in enumerate(d.train_loader, 0):
            inputs, labels = data
            labels.unsqueeze_(1)
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
            labels = labels.cuda()
            # inputs, labels = inputs.cuda(), labels.cuda()
            # print(labels)
            optimiser.zero_grad()
            outputs = cnet(inputs)
            # print(outputs)
            # print(outputs, torch.min(outputs), torch.max(outputs))
            loss_out = loss(outputs, labels)

            loss_out.backward()
            optimiser.step()
            running_loss += loss_out.item()
            total_train_loss += loss_out.item()
            # print(outputs, labels)
            predicted = torch.Tensor([yesno(outputs[v]) for v in range(0, len(outputs))]).long()

            labels = labels.cpu().long()
            for lb in range(len(labels)):
                if labels[lb] == 0:
                    run_counter[0] += 1
                    counter[0] += 1
                    # print("LB 0:", 1 if predicted[lb] == labels[lb] else 0)
                    correct[0] += 1 if predicted[lb] == labels[lb] else 0
                    total_correct[0] += 1 if predicted[lb] == labels[lb] else 0
                elif labels[lb] == 1:
                    run_counter[1] += 1
                    counter[1] += 1
                    # print("LB1:", predicted[lb], labels[lb], predicted[lb].eq(labels[lb]).item())
                    correct[1] += 1 if predicted[lb] == labels[lb] else 0
                    total_correct[1] += 1 if predicted[lb] == labels[lb] else 0

            # total_correct[0] += correct[0]
            # total_correct[1] += correct[1]
            itot += 1
            if (i + 1) % print_every == 0:
                # print(print_every, itot)
                n = 100 * (total_correct[0] / counter[0]) if counter[0] != 0 else 0.0
                y = 100 * (total_correct[1] / counter[1]) if counter[1] != 0 else 0.0
                print("Epoch {}, {:d}% run_loss: {:.4f} run acc: {}/{} ({:.2f}%) \ttot_loss: {:.4f} N_acc: {}/{} "
                      "({:.2f}%) Y_acc: {}/{} ({:.2f}%) "
                      "took {:.2f}s"
                      .format(epoch + 1, int(100 * (i + 1) / num_batches), running_loss / print_every, correct[0]+correct[1],
                              run_counter[0]+run_counter[1], 100 * ((correct[0]+correct[1]) / (run_counter[0] + run_counter[1])),
                              total_train_loss / itot, total_correct[0], counter[0], n,
                              total_correct[1], counter[1], y,
                              (time.time() - start_time)))
                correct = [0, 0]
                running_loss = 0.0
                run_counter = [0, 0]
                start_time = time.time()
        correct = [0, 0]
        total_val_loss = 0
        cnet.eval()
        count = [0, 0]
        for inputs, labels in d.val_loader:
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
            labels = labels.cuda()
            val_outputs = cnet(inputs)
            # print(labels)
            val_loss = loss(val_outputs, labels)

            total_val_loss += val_loss.item()
            predicted = torch.Tensor([yesno(val_outputs[v]) for v in range(0, len(val_outputs))]).long()
            # print(predicted)
            # print("Pred:", predicted)

            # print("EQ:", predicted.eq(labels.cpu().long()))
            labels = labels.cpu().long()
            for lb in range(len(labels)):
                if labels[lb] == 0:
                    count[0] += 1
                    # print("LB 0:", predicted[lb], labels[lb], predicted[lb].eq(labels[lb]).item())
                    correct[0] += 1 if predicted[lb] == labels[lb] else 0
                elif labels[lb] == 1:
                    count[1] += 1
                    # print("LB1:", predicted[lb], labels[lb], predicted[lb].eq(labels[lb]).item())
                    correct[1] += 1 if predicted[lb] == labels[lb] else 0

        tot = len(d.val_loader.dataset)
        tota = len(d.val_loader)
        n = 100 * (correct[0] / count[0]) if count[0] != 0 else 0.0
        y = 100 * (correct[1] / count[1]) if count[1] != 0 else 0.0
        yn = 100 * ((correct[1] + correct[0]) / (count[1] + count[0])) if (count[1] + count[0]) != 0 else 0.0
        print("Validation loss = {:.6f} n_Accuracy = {}/{} ({:.6f}%) y_Accuracy = {}/{} ({:.6f}%) Accuracy = {}/{} ({:.6f}%)"
              .format(total_val_loss / tota, correct[0], count[0], n, correct[1], count[1],
                      y, (correct[1] + correct[0]), (count[1] + count[0]), yn))

        print("Epoch and Validation took {:.2f}s".format(time.time() - s_t))
    print("Training finished, took {:.2f}s".format(time.time() - s_time))


def test(cnet, d):
    cnet.eval()
    total = 0
    correct = [0, 0]
    t_loss = 0.0
    count = [0, 0]
    with torch.no_grad():
        for inputs, target in d.test_loader:
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
            target = target.cuda()
            test_outputs = cnet(inputs)

            tt = F.binary_cross_entropy(test_outputs, target, reduce=False)
            t_loss += torch.sum(tt)
            predicted = torch.Tensor([yesno(test_outputs[v]) for v in range(0, len(test_outputs))]).long()

            target = target.cpu().long()
            for lb in range(len(target)):
                if target[lb] == 0:
                    count[0] += 1
                    print("LB 0:", predicted[lb], target[lb], 1 if predicted[lb] == target[lb] else 0)
                    correct[0] += 1 if predicted[lb] == target[lb] else 0
                elif target[lb] == 1:
                    count[1] += 1
                    print("LB1:", predicted[lb], target[lb], 1 if predicted[lb] == target[lb] else 0)
                    correct[1] += 1 if predicted[lb] == target[lb] else 0

    total = len(d.test_loader.dataset)
    t_loss /= total

    print('\nTest set: Average loss: {:.4f}, n_Accuracy: {}/{} ({:.4f}%) y_Accuracy: {}/{} ({:.4f}%)\n'
          .format(t_loss, correct[0], count[0], 100 * (correct[0] / count[0]), correct[1], count[1],
                  100 * (correct[1] / count[1])))


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
            net = torch.nn.DataParallel(Siamese.Siamese(**args), device_ids=dids).float().cuda()
        else:
            net = Siamese.Siamese(**args).float()

        d_prep = dp.DataPrep()
        train_cnn(net, d=d_prep, epochs=5, learn_rate=learning_rate)
        test(net, d_prep)
