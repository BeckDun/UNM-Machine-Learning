import task1
import task2
import matplotlib.pyplot as plt

datasets = task1.pre_process_data()

results = {}

for ticker in datasets:
    print(f"\n===== {ticker} =====")

    data = datasets[ticker]

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    model = task2.RNN(input_size=1)

    result = task2.train_model(model, X_train, y_train, X_test, y_test)

    print(f"Final Train Loss: {result['train_losses'][-1]}")
    print(f"Test Loss:        {result['test_losses'][-1]:.6f}")
    print(f"Training Time:    {result['training_time']:.2f} sec")

    results[ticker] = result

for ticker in results:
    train_losses = results[ticker]["train_losses"]
    test_losses = results[ticker]["test_losses"]

    plt.figure()  # new plot for each stock

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")

    plt.title(f"{ticker} Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid()

plt.show()