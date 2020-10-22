#using Flux
#using Images
#using Base.iterators:repeated
using Random
using LinearAlgebra
using MLDatasets

#https://github.com/nmheim/NeuralNetworks

struct W
	b
	w
end

struct Network
	sizes	
	num_layers
	w_and_b
end

#learning_rate = 1

#functions

function feed_forward(a,w_and_b)
	a = map((w,b)->sigma((w*a)+b),w_and_b.w,w_and_b.b)
	return a
end

function sigma(x)
	return 1/(1+exp(-x))
end



function sigma_prime(x)
	return sigma(x)*(1-sigma(x))
end

function sgd(train_data,epochs,mini_batch_size,eta,w_and_b,test_data=0)#,test_labels=0)
	if test_data != 0
		n_test = length(test_data)
	end
	for epoch=1:epochs
		n = length(train_data)
		#println([0:mini_batch_size:n])
		mini_batches = batch(train_data,mini_batch_size)
	
		new_bt = map(min_bt->updating_mini_batch(min_bt,eta,w_and_b),mini_batches)	
		println("ok")
		if test_data != 0
			println("Epoch ",epoch,":",evaluate(test_data),"/",n_test)
		else
			println("Epoch ",epoch,"complited")
		end
	end
end


function batch(x,bach_size)
	tmp = []
	for i=0:bach_size:(length(x)-bach_size)
		push!(tmp,x[i+1:i+bach_size])#updating_mini_batch(x[i+1:i+bach_size],eta,w_and_b)
	end
	return tmp
end


function updating_mini_batch(mini_batch,eta,w_and_b)
	#updating 'w' and 'b' 
	noble_b = map(x->zeros(1,length(x)),w_and_b.b) 
	noble_w = map(x->zeros(1,length(x)),w_and_b.w)

	for i=1:length(mini_batch)
		delta_b,delta_w = backprop(mini_batch[i][1],mini_batch[i][2],w_and_b)
		nabla_b = map((n_b,d_b)->n_b+d_b,noble_b,delta_b)
		nabla_w = map((n_w,d_w)->n_w+d_w,noble_w,delta_w)
		w_and_b.w = map((w,n_w)->w-(eta/length(mini_batch))*n_w,w_and_b.w,nable_w)
		w_and_b.b = map((b,n_b)->b-(eta/length(mini_batch))*n_b,w_and_b.b,nable_b)
	end
	#return 1
end

function backprop(x,y,w_and_b,num_layers=3)
	noble_b = map(x->zeros(1,length(x)),w_and_b.b) 
	noble_w = map(x->zeros(1,length(x)),w_and_b.w)
	activation = reshape(x,784)
	activations =[] #[reshape(x,784)]
	zs = []
	for i=1:length(w_and_b.w)
		
		# DimensionMismatch
		#z = dot(w_and_b.w[i][:,:],activation) + w_and_b.b[i]
		######################
		z = sum(w_and_b.w[i][:,:]*reshape(activation,784) + w_and_b.b[i])
		push!(zs,z)
		activation = sigma(z)

		push!(activations,activation)
		
		println("ok")
	end
	delta = cost_derivative(activations[length(activations)],y)*sigma_prime(zs[length(zs)])
	nabla_b[length(nabla_b)] = delta
	nabla_w[length(nabla_b)] = dot(delta,transpose(activations[length(activations)-1]))

	for i = 3:num_layers
		z = zs[length(zs)-i]
		sp = sigma_prime(z)
		delta = (transpose(w_and_b.w[length(w_and_b.w)-i+1]*delta))*sp
		nabla_b[length(nabla_b)-i] = delta
		nabla_w[length(nabla_w)-i] = delta*transpose(activations[length(activations)-i-1])
	end


	return (nabla_b,nabla_w)
end

function cost_derivative(output_active,y)
	return output_avtive-y
end

function evaluate(test_data,w_and_b)
	function arg_max(x)
		tmp = x[0]
		for h=1:length(x)
			if tmp<x[h]
				tmp = x[h]
			end
		end
		return tmp
	end

	test_result = map((x,y)->(arg_max(feed_forward(x,w_and_b)),y),test_data)
	#True = 1, False = 0
	return cumsum(map((x,y)->x==y,test_result),dims=1)[length(map((x,y)->x==y,test_result))]
end

function zipping(sample,label)
	a = []
	for i=1:length(label)
		push!(a,(sample[:,:,i],label[i]))	
	end
	return a
end

#Initiating NN

size = [784,30,10]
w_and_b = W(map(x->rand(Float64,x),size[2:length(size)]),map((x,y)->rand(Float64,(x,y)),size[2:length(size)],size[1:length(size)-1]))
#new_tmp = map(x->rand(Float64,x),size[2:length(size)][1])
#println()
#eachcol

#net = Network(size,length(size),w_and_b)

training_x,train_y = MNIST.traindata()
test_x,test_y = MNIST.testdata()

#println(length(test_x))


println("<============================>")
#println()
#println(zipping(training_x,train_y)[1])
#tmp = zipping(training_x,train_y)

#p = MNIST.traintensor(Float32)#MNIST.convert2features(MNIST.traintensor())
#@view p
#println(typeof(reshape(training_x,60000,784)))






#sgd(mapping(reshape(training_x,784,length(train_y)),train_y),30,10,3.0,w_and_b,mapping(reshape(test_x,784,length(test_y)),test_y))

sgd(zipping(training_x,train_y),30,10,3.0,w_and_b,zipping(test_x,test_y))




