from src.model import CNet, train, test
from src.utils import load_cifar10

if __name__ == '__main__':
    trainloader, valloader, testloader = load_cifar10(partition_id=0, num_clients=10)
    net = CNet().to("cpu")

    for epoch in range(10):
        train(net, trainloader, 1)
        loss, accuracy = test(net, valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    loss, accuracy = test(net, testloader)
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
