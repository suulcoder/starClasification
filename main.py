import csv

"""

###########################################################
# Proyecto Física - Radiación de cuerpo negro
###########################################################

# Este programa permitirá analizar radiación de cuerpo
# negro de los datos tomados por el observatorio de la
# universidad de hawai

# Reference: http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import leastsq

h = 4.135e-15
c = 3e8
k = 8.617e-5
wien = 2897.6

class BlackBody(object):
	"""docstring for BlackBody"""
	def __init__(self, file,Star_type):
		self.type = Star_type
		self.name = file
		flux = []
		wavelength = []
		error = []
		file = open(file + '.txt', "r")
		for line in file:
			if (line[0] != '#'):
				data_set = []
				data_row = line.replace('\n','').split(' ')
				for data_column in data_row:
					if(data_column != ''):
						data_set.append(data_column)
				if(float(data_set[1])>-100):
					for i in range(len(data_set)):
						if(i == 0):
							wavelength.append(float(data_set[i]))
						if(i == 1):
							flux.append(float(data_set[i]))
						if(i == 2):
							error.append(float(data_set[i]))
		self.flux = np.array(flux)
		self.wavelength = np.array(wavelength)
		self.error = np.array(error)

	def plot(self):
		plt.scatter(self.wavelength, self.flux)
		plt.axis([0.7,2.3,0.1e-10,1.5e-10])
		plt.xlabel("wavelength ")
		plt.ylabel("Flux")
		plt.show()

	def func(x):
	    return (2*h*x**3)/((np.exp(h*x/(k))) * c**2)

	def dy_forward(self,y,h):
	    N = len(y)
	    dy = np.zeros(N)
	    for k in range(N-1):
	        dy[k] = (y[k+1]-y[k])/h
	    return dy

	def funcion(self,x, p):
		return p[0]*(1/(np.exp(p[1]*x)-1))*(x**3)

	def derivate(self):
		# LISTAS CON LOS DATOS X  y Y
		datos_x = self.wavelength
		datos_y = self.flux

		# Ahora se trata de ajustar estos datos una función
		# modelo tipo exponencial  A*e**(B*x))

		#  Defino la funcion de residuos
		def residuos(p, y, x):
		    A, B = p
		    error = y - A*(1/(np.exp(B*x)-1))*(x**3)
		    return error

		# Parámetros iniciales

		# Si estos se alejan mucho del valor real
		# la solución no convergerá
		p0 = [5,10]

		# hacemos  el ajuste por minimos cuadrados
		ajuste = leastsq(residuos, p0, args=(datos_y, datos_x))

		# El resultado es una lista, cuyo primer elemento es otra
		# lista con los parámetros del ajuste.

		# Ahora muestro los datos y el ajuste gráficamente

		plt.plot(datos_x, datos_y, 'o')  # datos

		# genero datos a partir del modelo para representarlo
		x1 = np.arange(0, datos_x.max(), 0.00001)  # array con muchos puntos de x
		y1 = self.funcion(x1, ajuste[0])           # valor de la funcion modelo en los x

		rmse = 0
		for n in range(len(self.wavelength)):
			rmse += (self.flux[n] - self.funcion(self.wavelength[n],ajuste[0]))**2

		rmse = rmse/len(self.wavelength)
		rmse = math.sqrt(rmse)

		self.derivate = self.dy_forward(y1,0.00001)

		lamda_max = []

		for n in range(len(self.derivate)):
			if(self.derivate[n]<0 and self.derivate[n-1]>0):
				lamda_max.append(x1[n])
				
		avg_lamda_max = 0
		for n in lamda_max:
			avg_lamda_max += n

		avg_lamda_max = avg_lamda_max/len(lamda_max)
		Temperature =  str((wien / avg_lamda_max))
		delta_t =  str(rmse * (wien / avg_lamda_max**2))		

		plt.plot(x1, y1, 'r-')
		plt.ylabel('Densidad de Flujo (W m-2 um-1)')
		plt.xlabel('Longitud de Onda (microns)')
		plt.title('Ajuste de funcion exponencial')
		plt.legend(('Datos', 'Ajuste exponencial'))
		plt.savefig(self.name + ".jpg")
		plt.clf()
		filenames = ['Lamda_MAX_(micrometros)','Delta_Lamda_(rmse)','Tempeartura_(Kelvin)','Delta_Temperatura','Tipo_de_Estrella','Archivo']
		with open('data.csv', mode='a') as csv_file:
		    writer = csv.DictWriter(csv_file, fieldnames=filenames)
		    writer.writerow({
		    	'Lamda_MAX_(micrometros)': str(avg_lamda_max),
		    	'Delta_Lamda_(rmse)': rmse,
		    	'Tempeartura_(Kelvin)': Temperature,
		    	'Delta_Temperatura': delta_t,
		    	'Tipo_de_Estrella': self.type,
		    	'Archivo': self.name})

for i in range(1,12):
	print("\n\n STAR TYPE G &" + str(i))
	myBlackBody = BlackBody("g" + str(i), Star_type='G')
	myBlackBody.derivate()
	print("\n\n STAR TYPE F &" + str(i))
	myBlackBody = BlackBody("f" + str(i), Star_type='F')
	myBlackBody.derivate()
	print("\n\n STAR TYPE M &" + str(i))
	myBlackBody = BlackBody("m" + str(i), Star_type='m')
	myBlackBody.derivate()
	print("\n\n STAR TYPE K &" + str(i))
	myBlackBody = BlackBody("k" + str(i), Star_type='K')
	myBlackBody.derivate()
	print("\n\n Dwarf TYPE L &" + str(i))
	myBlackBody = BlackBody("L_Dwarf" + str(i), Star_type='L_Dwarf')
	myBlackBody.derivate()