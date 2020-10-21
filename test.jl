#using Flux
#using Images
#using Base.iterators:repeated
using Random
using LinearAlgebra
using MLDatasets

#https://github.com/nmheim/NeuralNetworks

struct Layer
	w
	b
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
		#mini_batches = map(x->train_data[x+1:x+mini_batch_size],[0:mini_batch_size:n])

	
		new_bt = map(min_bt->updating_mini_batch(train_data[min_bt+1:min_bt+mini_batch_size,:],eta,w_and_b),[0:mini_batch_size:n])	
		println("ok")
		if test_data != 0
			println("Epoch ",epoch,":",evaluate(test_data),"/",n_test)
		else
			println("Epoch ",epoch,"complited")
		end
	end
end


function batch(x,bath_size)

end



function updating_mini_batch(mini_batch,eta,w_and_b,)
	#updating 'w' and 'b' 
	noble_b = map(x->zeros(1,length(x)),w_and_b.b) 
	noble_w = map(x->zeros(1,length(x)),w_and_b.w)

	for i=1:length(mini_batch)
		delta_b,delta_w = backprop(minibatch[i][0],minibatch[i][1],w_and_b)
		nabla_b = map((n_b,d_b)->n_b+d_b,noble_b,delta_b)
		nabla_w = map((n_w,d_w)->n_w+d_w,noble_w,delta_w)
		w_and_b.w = map((w,n_w)->w-(eta/length(mini_batch))*n_w,w_and_b.w,nable_w)
		w_and_b.b = map((b,n_b)->b-(eta/length(mini_batch))*n_b,w_and_b.b,nable_b)
	end
	return 1
end

function backprop(x,y,w_and_b,num_layers)
	noble_b = map(x->zeros(1,length(x)),w_and_b.b) 
	noble_w = map(x->zeros(1,length(x)),w_and_b.w)
	activation = x
	activations = [x]
	zs = []
	for i=1:length(w_and_b.w)
		z = w*activation + w_and_b.b[i]
		zs[i] = z
		activation = sigmoid(z)
		activasions[i+1] = activation
	end
	delta = cost_derivative(activations[length(activations)],y)*sigmoid_prime(zs[length(zs)])
	nabla_b[length(nabla_b)] = delta
	nabla_w[length(nabla_b)] = delta*transpose(activations[length(activations)-1])

	for i = 3:num_layers
		z = zs[length(zs)-i]
		sp = sigmoid_prime(z)
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

function mapping(sample,label)
	a = []
	for i=1:length(label)
		push!(a,(sample[i,:],label[i]))	
	end
	return a
end

#Initiating NN

size = [784,30,10]
w_and_b = W(map(x->rand(Float64,(x,1)),size[2:length(size)]),map((x,y)->rand(Float64,(x,y)),size[2:length(size)],size[length(size)]))

#eachcol

#net = Network(size,length(size),w_and_b)

training_x,train_y = MNIST.traindata()
test_x,test_y = MNIST.testdata()

#println(length(test_x))


println("<============================>")
println()



#p = MNIST.traintensor(Float32)#MNIST.convert2features(MNIST.traintensor())
#@view p
#println(typeof(reshape(training_x,60000,784)))
#println(length(train_y))
#println(length(reshape(training_x,length(train_y),784)[1,:]))

#println(length(mapping(reshape(training_x,length(train_y),784),train_y)))
#println(map([x,y],reshape(training_x,length(train_y),784),train_y))

sgd(mapping(reshape(training_x,784,length(train_y)),train_y),30,10,w_and_b,mapping(reshape(test_x,784,length(test_y)),test_y))




