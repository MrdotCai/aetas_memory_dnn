import methods.data_reader.mnist_mix_color as mnist_reader


def data_generate(dataset_name="mnist"):
    return mnist_reader.read_data_sets('./data_set')

def train_batch_generator(batch_size=10):
    mnist_data = data_generate()
    for batch_start in range(0, 600, batch_size):
        yield mnist_data.train.data[batch_start:batch_start+batch_size], mnist_data.train.target[batch_start:batch_start+batch_size], mnist_data.train.color[batch_start:batch_start+batch_size]

if __name__=="__main__":
    # mnist_data = data_generate()
    # print(mnist_data.train.color[-10:])
    # print(mnist_data.train.target[-10:])
    # print(mnist_data.train.data[0].shape)
    gt = train_batch_generator()
    print(next(gt))