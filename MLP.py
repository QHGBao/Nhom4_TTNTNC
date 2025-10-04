import numpy as np

class MLPTextClassifier:
    def __init__(self, vocab, embed_dim=4, hidden_neurons=2, output_dim=3, activation="relu", lr=0.1):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.hidden_neurons = hidden_neurons
        self.output_dim = output_dim
        self.lr = lr
        self.activation_name = activation

        # Khởi tạo tất cả tham số = 0.5
        self.Embedding = np.full((self.vocab_size, embed_dim), 0.5)  # 5x4
        self.W1 = np.full((embed_dim, hidden_neurons), 0.5)  # 4x2
        self.b1 = np.full((hidden_neurons,), 0.5)            # 2
        self.W2 = np.full((hidden_neurons, output_dim), 0.5) # 2x3
        self.b2 = np.full((output_dim,), 0.5)                # 2

    def activation(self, x):
        if self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == "relu":
            return np.maximum(0, x)

    def activation_deriv(self, x):
        if self.activation_name == "sigmoid":
            sig = self.activation(x)
            return sig * (1 - sig)
        elif self.activation_name == "relu":
            return (x > 0).astype(float)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / (np.sum(e_x) + 1e-10)

    def forward(self, word_idx):
        # Embedding lookup
        x_embed = self.Embedding[word_idx]  # vector 1x4

        # Hidden
        z1 = np.dot(x_embed, self.W1) + self.b1
        h = self.activation(z1)

        # Output
        z2 = np.dot(h, self.W2) + self.b2
        y_hat = self.softmax(z2)

        # In tham số forward
        print("=== Forward ===")
        print(f"x_embed: {x_embed}")
        print(f"z1: {z1}")
        print(f"h: {h}")
        print(f"z2: {z2}")
        print(f"y_hat: {y_hat}")

        return x_embed, z1, h, z2, y_hat

    def backward(self, word_idx, x_embed, z1, h, z2, y_hat, y_true):
        loss = -np.sum(y_true * np.log(y_hat + 1e-10))

        # Grad output
        dz2 = y_hat - y_true
        dW2 = np.outer(h, dz2)
        db2 = dz2

        # Grad hidden
        dz1 = np.dot(dz2, self.W2.T) * self.activation_deriv(z1)
        dW1 = np.outer(x_embed, dz1)
        db1 = dz1

        # Grad embedding (chỉ update vector tại word_idx)
        dEmbed = np.dot(dz1, self.W1.T)

        # In gradient
        print("=== Backward ===")
        print(f"dz2: {dz2}")
        print(f"dW2:\n{dW2}")
        print(f"db2: {db2}")
        print(f"dz1: {dz1}")
        print(f"dW1:\n{dW1}")
        print(f"db1: {db1}")
        print(f"dEmbed: {dEmbed}")
        print(f"Loss: {loss}")

        # Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.Embedding[word_idx] -= self.lr * dEmbed

        # In tham số sau cập nhật
        print("=== Updated Parameters ===")
        print(f"W1:\n{self.W1}")
        print(f"b1: {self.b1}")
        print(f"W2:\n{self.W2}")
        print(f"b2: {self.b2}")
        print(f"Embedding:\n{self.Embedding}")
        print("---------------------------")

        return loss

    def train(self, data, labels, epochs=1):
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch} =====")
            for idx, word in enumerate(data, start=1):
                print(f"\n--- Sample {idx}: {word} ---")
                word_idx = self.vocab.index(word)
                y_true = labels[idx-1]

                # Forward
                x_embed, z1, h, z2, y_hat = self.forward(word_idx)

                # Backward
                self.backward(word_idx, x_embed, z1, h, z2, y_hat, y_true)


# Vocab và training set
vocab = ["He", "is", "reading", "a", "book"]
train_set = ["He", "is", "reading", "a", "book"]

# Ví dụ nhãn (giả định 2 lớp → one-hot)
labels = [
    np.array([1,0]),
    np.array([0,1]),
    np.array([1,0]),
    np.array([0,1]),
    np.array([1,0])
]

mlp = MLPTextClassifier(vocab, activation="relu", lr=0.1)
mlp.train(train_set, labels, epochs=1)
