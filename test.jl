using Random
using LinearAlgebra
using MLDatasets
using StatsBase


struct Layer
	b::Vector
	W::Matrix
end

struct Network
	layers::Vector{Layer}
end

function Layer(in::Int,out::Int)
	W = rand(out,in)
	b = rand(out)
	Layer(b,W)
end


function (n::Network)(x::Vector)
	v = [x]
	zs = []
	for i in n.layers
		push!(zs,i.W*x+i.b)
		push!(v,i(x))
		x = v[end]
	end
	return zs,v
end


function (l::Layer)(x::Vector)
	tmp = sigma.(l.W*x+l.b)
	return tmp
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
		#mini_batches = train_data[shuffle(1:end)]
		mini_batches = random_batch(train_data,mini_batch_size)
		map(mini_batch->updating_mini_batch(random_batch(train_data,mini_batch_size),eta,model),train_data)

		if test_data != 0
			println("Epoch ",epoch,":",evaluate(test_data,model),"/",n_test)
		else
			println("Epoch ",epoch,"complited")
		end
	end
end


function random_batch(x::Vector,batch_size::Int)
	indx = sample(1:Base.size(x,1),batch_size,replace = false)
	return x[indx]
end


function updating_mini_batch(mini_batch,eta,model)
	#updating 'w' and 'b' 
	noble_b = map(x->zeros(size(x.b)),model.layers) 
	noble_w = map(x->zeros(size(x.W)),model.layers)
	for i=1:length(mini_batch)
		delta_b,delta_w = backprop(mini_batch[i][1],mini_batch[i][2],model,length(model.layers)+1)
		all_W,all_b = map(l->l.W,model.layers),map(l->l.b,model.layers)
		new_W = all_W-(eta/length(mini_batch))*delta_w
		new_b = all_b-(eta/length(mini_batch))*delta_b
		for i=1:length(model.layers) 
			model.layers[i].W .= new_W[i]
			model.layers[i].b .= new_b[i]
		end
			
	end
end


function backprop(x::Vector,y::Int,model::Network,num_layers=3)
	noble_b = map(x->zeros(size(x.b)),model.layers) 
	noble_w = map(x->zeros(size(x.W)),model.layers)
	zs,activations = model(x) 

	delta = cost_derivative(activations[end],y).*map(b->sigma_prime(b),zs[end])

	noble_b[end] .= delta 		
	noble_w[end] .= delta*activations[end-1]'
	
	for i = length(model.layers)-1:-1:1 	
		z = zs[i]
		sp = sigma_prime.(z)
		layer = model.layers[i+1]
		W,b = layer.W,layer.b
		delta = (W'*delta).*sp
		noble_b[i] .= delta#reshape(delta,length(delta))
		noble_w[i] .= delta*activations[i]'
	end

	return noble_b,noble_w
end


function cost_derivative(output_active,y)
	# making number y into array of zeros where y-th elem is 1 
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


function zipping(sample,label::Array)
	a = []
	for i=1:length(label)
		push!(a,(reshape(sample[:,:,i],length(sample[:,:,i])),label[i]))	
	end
	return a
end


#Initiating NN

training_x,train_y = MNIST.traindata()
test_x,test_y = MNIST.testdata()
tr_x = MNIST.traintensor(Float64)
println("<============================>")

# Updated code


model = Network(
	[Layer(784,30),
	Layer(30,10)]
)


data = zipping(training_x,train_y)
tst = zipping(test_x,test_y)
#evaluate(tst,model)
#display(train_y)
sgd(data,30,10,3.0,model,tst)
