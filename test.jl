#using Flux
#using Images
#using Base.iterators:repeated
using Random
using LinearAlgebra
using MLDatasets
using StatsBase


mutable struct Layer
	b::Vector
	W::Matrix
end

mutable struct Network
	layers::Vector{Layer}
end

function Layer(in::Int,out::Int)
	W = rand(out,in)
	b = rand(out)
	Layer(b,W)
end


function (n::Network)(x::Vector)
	v = []
	for i in n.layers
		v = i(x)
		x = v
	end
	return x
end


function (l::Layer)(x::Vector)
	tmp = sigma.(l.W*x+l.b)
	return tmp
end


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


function sgd(train_data,epochs,mini_batch_size,eta,model,test_data=0)
	if test_data != 0
		n_test = length(test_data)
	end
	for epoch=1:epochs
		train_data = train_data[shuffle(1:end)]
		#mini_batches = random_batch(train_data,mini_batch_size)
	
		new_bt = map(min_bt->updating_mini_batch(random_batch(train_data,mini_batch_size),eta,model),train_data)	
		println("ok")
		if test_data != 0
			println("Epoch ",epoch,":",evaluate(test_data,model),"/",n_test)
		else
			println("Epoch ",epoch,"complited")
		end
	end
end


function random_batch(x::Array,batch_size::Int)
	indx = sample(1:Base.size(x)[1],batch_size,replace = false)
	return x[indx]
	#a = sample(x[:,indx]
	#tmp = []
	#for i=1:batch_size:(length(x)-batch_size)
	#	push!(tmp,x[i:i-1+batch_size])
	#end
	#return tmp
end


function updating_mini_batch(mini_batch,eta,model)
	#updating 'w' and 'b' 
	noble_b = map(x->zeros(size(x.b)),model.layers) 
	noble_w = map(x->zeros(size(x.W)),model.layers)
	nabla_b = []
	nabla_w = []
	for i=1:length(mini_batch)
		delta_b,delta_w = backprop(mini_batch[i][1],mini_batch[i][2],model,length(model.layers)+1)
		nabla_b = noble_b+delta_b#map((n_b,d_b)->n_b+d_b,noble_b,delta_b)
		#println(size(delta_w[2]))
		#display(size(noble_w[2]))
		nabla_w = noble_w+delta_w#map((n_w,d_w)->n_w+d_w,noble_w,delta_w)
		model.layers = map((l,n_w,n_b)->Layer(l.b-(eta/length(mini_batch))*n_b,l.W-(eta/length(mini_batch)*n_w)),model.layers,nabla_w,nabla_b)
		#map((w,n_w)->w-(eta/length(mini_batch))*n_w,w_and_b.w,nable_w)
		#model.layers.b = #map((b,n_b)->b-(eta/length(mini_batch))*n_b,w_and_b.b,nable_b)
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


function backprop(x,y,model,num_layers=3)
	noble_b = map(x->zeros(size(x.b)),model.layers) 
	noble_w = map(x->zeros(size(x.W)),model.layers)
	activation = reshape(x,1,length(x))#convert(Array{Float64},reshape(x,1,784))
	activations =[activation]
	zs = []
	
	for i=1:size(model.layers)[1]
		
		# DimensionMismatch
		#z = dot(w_and_b.w[i][:,:],activation) + w_and_b.b[i]
		######################
		#z = dot_product(model.layers[i].W,activation,model.layers[i].b)
		z = sum(model.layers[i].W.*activations[end],dims = 2)+model.layers[i].b

		push!(zs,z)
		activation = map(g->sigma(g),z)
	
		push!(activations,reshape(activation,1,length(activation)))
		
	end
	
	#println(Base.size(cost_derivative(activations[length(activations)],y)))
	reshape_size = Base.size(map(b->sigma_prime(b),zs[length(zs)]))
	
	delta = cost_derivative(activations[length(activations)],y).*reshape(map(b->sigma_prime(b),zs[length(zs)]),1,reshape_size[1])

	#println(Base.size(delta))
	#println(delta)
	#println(typeof(noble_b[length(noble_b)]))
	delta = reshape(delta,length(delta))

	noble_b[end] = delta 	
	#println("ok")
	#println(Base.size(noble_w[length(noble_b)]))
	#display(size(delta))
	noble_w[end] = activations[end-1].*delta
	#delta.*transpose(activations[length(activations)-1])
	#transpose(activations[length(activations)-1])

	#println(length(zs[1]))
	for i = 2:num_layers-1
		z = zs[length(zs)-i+1]
		sp = sigma_prime.(z)
		delta = sum(transpose(model.layers[length(model.layers)-i+2].W.*delta),dims = 2).*sp
		
		#println(size(noble_w[end-i+1]))
		noble_b[end-i+1] = reshape(delta,length(delta))
		noble_w[end-i+1] = activations[end-i].*delta
		#transpose(activations[length(activations)-i]).*reshape(delta,1,length(delta))
	end
	#display("Backprop Ok")
	return noble_b,noble_w
end


function cost_derivative(output_active,y)
	# making numer y into array of zeros where y-th elem is 1 
	new_y = zeros(Base.size(output_active))
	new_y[y+1] = 1
	return output_active-new_y
end


function evaluate(test_data,model)
	test_result = map(x->(findmax(model(reshape(x[1],length(x[1]))))[2],x[2]),test_data)
	#display(test_result)
	#True = 1, False = 0
	return cumsum(map(x->x[1]==x[2],test_result),dims=1)[length(map(x->x[1]==x[2],test_result))]
end


function zipping(sample::Matrix,label::Array)
	a = Vector[]
	for i=1:length(label)
		push!(a,(sample[:,:,i],label[i]))	
	end
	return a
end


#Initiating NN

#w_and_b = Layer(map(x->rand(Float64,x),size[2:length(size)]),map((x,y)->rand(Float64,(x,y)),size[2:length(size)],size[1:length(size)-1]))


training_x,train_y = MNIST.traindata()
test_x,test_y = MNIST.testdata()
tr_x = MNIST.traintensor(Float64)
println("<============================>")

#sgd(zipping(training_x,train_y),30,10,3.0,w_and_b,zipping(test_x,test_y))
# Old code

# Updated code


model = Network(
	[Layer(784,30),#(784,30),
	Layer(30,10)]#(30,10)
)


data = zipping(tr_x,train_y)
tst = zipping(test_x,test_y)
#evaluate(tst,model)
#display(train_y)
sgd(data,30,10,3.0,model,tst)
