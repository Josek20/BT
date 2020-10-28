#using Flux
#using Images
#using Base.iterators:repeated
using Random
using LinearAlgebra
using MLDatasets

#https://github.com/nmheim/NeuralNetworks

mutable struct Layer
	b::Array
	W::Array
end

mutable struct Network
	layers::Vector{Layer}
	#input_layer::Layer
	#output_layer::Layer
end

function Layer(in::Int,out::Int)
	W = rand(out,in)#{Matrix}
	b = rand(out,1)#{Vector}
	Layer(b,W)
end

function (n::Network)(x::Vector)
	v = []
	for i in n.layers
		v = i(x)
		x = reshape(v,length(v))
	end
	return x
end

function (l::Layer)(x::Vector)
	tmp = map(s->sigma(s),l.W*x+l.b)
	#println(tmp)
	return tmp
end

#learning_rate = 1

#functions

function feed_forward(a,w_and_b)
	# suppose to be dot product
	a = map((w,b)->sigma(dot(w,a)+b),w_and_b.w,w_and_b.b)
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


function batch(x,batch_size)
	tmp = []
	for i=1:batch_size:(length(x)-batch_size)
		push!(tmp,x[i:i-1+batch_size])
	end
	return tmp
end


function updating_mini_batch(mini_batch,eta,w_and_b)
	#updating 'w' and 'b' 
	noble_b = map(x->zeros(1,length(x)),w_and_b.b) 
	noble_w = map(x->zeros(1,length(x)),w_and_b.w)
	nabla_b = []
	nabla_w = []
	for i=1:length(mini_batch)
		delta_b,delta_w = backprop(mini_batch[i][1],mini_batch[i][2],w_and_b)
		println("ok")
		nabla_b = map((n_b,d_b)->n_b+d_b,noble_b,delta_b)
		nabla_w = map((n_w,d_w)->n_w+d_w,noble_w,delta_w)
		w_and_b.w = map((w,n_w)->w-(eta/length(mini_batch))*n_w,w_and_b.w,nable_w)
		w_and_b.b = map((b,n_b)->b-(eta/length(mini_batch))*n_b,w_and_b.b,nable_b)
	end
	#return 1
end


function dot_product(w,x,b=0)
	rmp = []
	#println("ok :",length(w))
	for i=1:Base.size(w)[1]
		#println(Base.size(w[i,:]))
		push!(rmp,dot(w[i,:],x))
	end
	return rmp+b
end


function backprop(x,y,w_and_b,num_layers=3)
	noble_b = map(x->zeros(Base.size(x)),w_and_b.b) 
	noble_w = map(x->zeros(Base.size(x)),w_and_b.w)
	activation = convert(Array{Float64},reshape(x,1,784))
	activations =[activation]
	zs = []
	for i=1:Base.size(w_and_b.w)[1]
		
		# DimensionMismatch
		#z = dot(w_and_b.w[i][:,:],activation) + w_and_b.b[i]
		######################
		z = dot_product(w_and_b.w[i],activation,w_and_b.b[i])
		push!(zs,z)
		activation = map(g->sigma(g),z)
	
		push!(activations,reshape(activation,1,length(activation)))
		
		#println("ok")
	end
	
	#println(Base.size(cost_derivative(activations[length(activations)],y)))
	reshape_size = Base.size(map(b->sigma_prime(b),zs[length(zs)]))
	delta = cost_derivative(activations[length(activations)],y).*reshape(map(b->sigma_prime(b),zs[length(zs)]),1,reshape_size[1])

	#println(Base.size(delta))
	println(delta)
	println(typeof(noble_b[length(noble_b)]))
	noble_b[length(noble_b)] = reshape(delta,length(delta))
	println("ok")
	#println(Base.size(noble_w[length(noble_b)]))
	noble_w[length(noble_b)] = dot_product(delta,activations[length(activations)-1])
	#illegal moves
	#transpose(activations[length(activations)-1])

	println("ok")
	for i = 3:num_layers
		z = zs[length(zs)-i]
		sp = sigma_prime(z)
		delta = (transpose(w_and_b.w[length(w_and_b.w)-i+1]*delta))*sp
		noble_b[length(noble_b)-i] = delta
		noble_w[length(noble_w)-i] = delta*transpose(activations[length(activations)-i-1])
	end


	return noble_b,noble_w
end

function cost_derivative(output_active,y)
	# making numer y into array of zeros where y-th elem is 1 
	new_y = zeros(Base.size(output_active))
	new_y[y+1] = 1
	return output_active-new_y
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

#size = [784,30,10]
#w_and_b = Layer(map(x->rand(Float64,x),size[2:length(size)]),map((x,y)->rand(Float64,(x,y)),size[2:length(size)],size[1:length(size)-1]))
#net = Network(size,length(size),w_and_b)
training_x,train_y = MNIST.traindata()
test_x,test_y = MNIST.testdata()
println("<============================>")
#sgd(zipping(training_x,train_y),30,10,3.0,w_and_b,zipping(test_x,test_y))
# Old code

# Updated code


model = Network(
	[Layer(2,3),#(784,30),
	Layer(3,1)]#(30,10)
)
x = randn(2)
# Still dosenot working((
y = model(x)
println(y)
@assert Base.size(y) == (1,)

