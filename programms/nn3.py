#Трёхслойная НС
#По Тарику Рашиду


import numpy
#библиотека scipy.special(сигмоида)
import scipy.special
#библиотека для графического отображения массивов
import matplotlib.pyplot
#пригодится позже
import time
from tqdm import tqdm
#две библиотеки неоюходимые для отслеживания прогресса при долгой отгрузке НС



#определение класса НС
class neuralNetwork:
    #инициализация НС
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #кол-во узлов в слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #матрицы весовых коэффициентов 
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        #коэффициент обучения
        self.lr = learningrate
        
        #сигмоида - функция активации 
        self.activation_function = lambda x: scipy.special.expit(x)

        pass
        
    #тренировка НС
    def train(self, inputs_list, targets_list):
        #входные --> массив
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin = 2).T
        
        #рассчёт входных сигналов для скрытого
        hidden_inputs = numpy.dot(self.wih, inputs)
        #рассчёт исходящих для скрытого
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #рассчёт входных для выходного
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #рассчитать исходящие для выходного
        final_outputs = self.activation_function(final_inputs)
        
        #ошибки выходного слоя = целевое - фактическое
        #далее ошибки распределяются пропорционально весам
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #обновление весов скрытый - выходной
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                       numpy.transpose(hidden_outputs))
        
        #Обновление весов входной-скрытый
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                       numpy.transpose(inputs))
        
        pass


    #опрос НС
    def query(self, inputs_list):
        #входные --> массив
        inputs = numpy.array(inputs_list, ndmin = 2).T
        
        #рассчёт входных сигналов для скрытого
        hidden_inputs = numpy.dot(self.wih, inputs)
        #рассчёт исходящих для скрытого
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #рассчёт входных для выходного
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #рассчитать исходящие для выходного
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

#узлы
input_nodes = 784 #всего 784 числа в численной записи изображения
hidden_nodes = 256 #можно поэкспериметировать 
output_nodes = 10 #всего 10 чисел (включая 0)

#коэффициент обучения
learning_rate = 0.2 #можно поэкспериметировать

#экземпляр НС
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)  

#тренировочный набор данных
training_data_file = open("C:\\NN\\data\\train60k.txt", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#тренировка НС

#эпохи
epochs = 7 #эксперементально

for e in range(epochs):
    #все записи тренировочного набора
    for record in tqdm(training_data_list, desc = "Эпоха " + str(e + 1) + " из " + str(epochs)):
        #список значений, используя запяту
        all_values = record.split(',')
        #масштабирование
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        #создать целевые выходные значения
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
        
#тестовый набор данных MNIST
test_data_file = open("C:\\NN\\data\\test10k.txt", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#Тестирование нейронной сети

#журнал оценок работы НС
scorecard = []

for record in test_data_list:
    #список значений, исполдьзуя запятую
    all_values = record.split(',')
    #правильный ответ - первое значение
    correct_label = int(all_values[0])
    #масштабирование
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #опрос НС
    outputs = n.query(inputs)
    #индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs)
    #оценка ответа(правильный ответ - будет оцениваться 1)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass

    pass
        
#показатель эффективности 
scorecard_array = numpy.asarray(scorecard)
eff = scorecard_array.sum() / scorecard_array.size
print('Эффективность = ',  eff)


#далее создание файла для прослежки динамики в зависимости от изменяемых параметров
result = open("C:\\NN\\Searching\\ResData.txt", 'a')
result.write('Эффективность: ' + str(eff) +  ' Скрытые узлы : ' + str(hidden_nodes) + ' Коэф: ' + str(learning_rate) + ' Эпохи: ' + str(epochs) + '\n')
result.close()