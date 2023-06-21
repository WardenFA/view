def my_format(fileinput, fileoutput):
	fin = open(fileinput, "r")
	fout = open(fileoutput, "a")
	for line in fin:
		l = line[:22].split()
		ans = l.index('1')
		st = line[24:]
		st = st.split()
		st = ",".join(st)
		fout.write(str(ans) + ',' + '0,' + st + '\n')
	fin.close(); fout.close()

	pass

my_format("C:\\NN\\data\\restest.txt", "C:\\NN\\data\\test10k.txt")

my_format("C:\\NN\\data\\restrain.txt", "C:\\NN\\data\\train60k.txt")