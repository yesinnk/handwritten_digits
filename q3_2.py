import torch
import torch.nn as nn
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.autograd import Variable
import data 

train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('a3digits.zip', 'data')


def process_data():
    train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('a3digits.zip', 'data')
    train_labels, test_labels = label_to_oneht(train_labels), label_to_oneht(test_labels)

    train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
    train_data, train_labels, test_data, test_labels = Variable(train_data), Variable(train_labels) \
        , Variable(test_data), Variable(test_labels)
    return train_data, train_labels, test_data, test_labels


class Net(nn.Module):
    def __init__(self, in_features=64, out_features=10):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 64, bias=True)
        self.predict = nn.Linear(64, 10, bias=True)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.predict(x))
        return x


def label_to_oneht(labels):
    tensors = []
    for i in labels:
        one_hot = [0] * 10
        one_hot[int(i)] = 1
        tensors.append(one_hot)
    return torch.tensor(tensors)


def save_model(net):
    address = "nnsave"
    torch.save(net.state_dict(), address)
    print("nn is save in :", address)


def load_model():
    the_model = Net()
    the_model.load_state_dict(torch.load("nnsave"))
    return the_model


def nnmain():
    train_data, train_labels, test_data, test_labels = process_data()
    net = Net()
    net.train()
    net = net.float()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2, momentum=0.5)
    loss_func = torch.nn.CrossEntropyLoss()
    prediction = net(train_data.float())
    for t in range(10000):
        optimizer.zero_grad()
        prediction = net(train_data.float())
        loss = loss_func(prediction, torch.max(train_labels, 1)[1])  # //prediction in first
        loss.backward()
        optimizer.step()
    result = [int(prediction[i].argmax()) for i in range(prediction.shape[0])]
    # print(result)
    count = 0

    for i in range(7000):
        if result[i] == torch.max(train_labels, 1)[1].tolist()[i]:
            count = count + 1
    print("MLP training accuray is ", count / 7000)

    final = 0
    with torch.no_grad():
        output = net(test_data.float())
        output = [int(output[i].argmax()) for i in range(output.shape[0])]
        for i in range(4000):
            if output[i] == torch.max(test_labels, 1)[1].tolist()[i]:
                final = final + 1
    print("MLP final accuracy is :", final / 4000)
    save_model(net)


def svmmain() -> svm.LinearSVC:
    train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('a2digits.zip', 'data')
    clf = svm.LinearSVC()
    clf.fit(train_data, train_labels)
    print("accuracy for SVM on test set is ", (clf.predict(test_data) == test_labels).sum() / 4000)
    return clf
    # print( "accuracy for SVM on train set is ",(clf.predict(train_data)==train_labels).sum()/7000)


def adaboostmain():
    bdt_SAMME = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=600,
        learning_rate=1.5,
        algorithm="SAMME")
    bdt_SAMME.fit(train_data, train_labels)
    return bdt_SAMME
    print("accuracy for ADABOOST with SAMME on test set is ",
          (bdt_SAMME.predict(test_data) == test_labels).sum() / 4000)


def main():
    nnmain()
    svmmain()
    adaboostmain()
    Assigments.a3.Model_Comparison.main()
    return


if __name__ == '__main__':
    main()