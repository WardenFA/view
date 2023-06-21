# converter_mnist.py
# raw binary MNIST to Keras text file

# target format:
# 0 0 1 0 0 0 0 0 0 0 ** 0 0 152 27 .. 0
# 0 1 0 0 0 0 0 0 0 0 ** 0 0 38 122 .. 0
#   10 vals at [0-9]    784 vals at [11-795]
# dummy ** seperator at [10] 

def generate(img_bin_file, lbl_bin_file,
            result_file, n_images):

  img_bf = open(img_bin_file, "rb")    # binary image pixels
  lbl_bf = open(lbl_bin_file, "rb")    # binary labels
  res_tf = open(result_file, "w")      # result text file

  img_bf.read(16)   # discard image header info
  lbl_bf.read(8)    # discard label header info

  for i in range(n_images):   # number images requested 
    # digit label first
    lbl = ord(lbl_bf.read(1))  # get label like '3' (one byte) 
    encoded = [0] * 10         # make one-hot vector
    encoded[lbl] = 1
    for i in range(10):
      res_tf.write(str(encoded[i]))
      res_tf.write(" ")  # like 0 0 0 1 0 0 0 0 0 0 

    res_tf.write("** ")  # arbitrary for readibility

    # now do the image pixels
    for j in range(784):  # get 784 vals for each image file
      val = ord(img_bf.read(1))
      res_tf.write(str(val))
      if j != 783: res_tf.write(" ")  # avoid trailing space 
    res_tf.write("\n")  # next image

  img_bf.close(); lbl_bf.close();  # close the binary files
  res_tf.close()                   # close the result text file

# ================================================================

def main():
  # change target file names, uncomment as necessary

  # make training data
  generate("C:\\NN\\data\\train-images.idx3-ubyte",
           "C:\\NN\\data\\train-labels.idx1-ubyte",
           "C:\\NN\\data\\restrain.txt",
           n_images = 60000)  # first n images

  # make test data
  generate("C:\\NN\\data\\t10k-images.idx3-ubyte",
           "C:\\NN\\data\\t10k-labels.idx1-ubyte",
           "C:\\NN\\data\\restest.txt",
           n_images = 10000)  # first n images

if __name__ == "__main__":
  main()