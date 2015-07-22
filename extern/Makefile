all: train_imgs test_imgs

bin: src
	mkdir -p bin/
	javac -d bin/ src/seadog/*.java

dat/train_imgs:
	mkdir -p dat/
	wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O dat/train_imgs.gz
	gzip -d dat/train_imgs.gz

dat/train_labels: 
	mkdir -p dat/
	wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O dat/train_labels.gz
	gzip -d dat/train_labels.gz

dat/test_imgs:
	mkdir -p dat/
	wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O dat/test_imgs.gz
	gzip -d dat/test_imgs.gz

dat/test_labels:
	mkdir -p dat/
	wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O dat/test_labels.gz
	gzip -d dat/test_labels.gz

train_imgs: bin dat/train_labels dat/train_imgs
	mkdir -p train_imgs/
	java -cp bin seadog.MNISTParser dat/train_labels dat/train_imgs train_imgs

test_imgs: bin dat/test_labels dat/test_imgs
	mkdir -p test_imgs/
	java -cp bin seadog.MNISTParser dat/test_labels dat/test_imgs test_imgs

clean:
	-rm -r bin
	-rm -r dat
	-rm -r train_imgs
	-rm -r test_imgs
