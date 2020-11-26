using Plots
using Random
using LinearAlgebra
using MLDatasets
using StatsBase

x_plot = []
y_plot = []
#write down into LaTex SGD

struct Layer
	b::Vector
	W::Matrix
end



struct Network
	layers::Vector{Layer}
	
	#inner contructor
	#Network{T}(ngenes::UInt) where T<:Num
end




function Layer(in::Int,out::Int)
	W = randn(out,in)
	b = randn(out)
	Layer(b,W)
end


function (n::Network)(x::Vector)
	active = [x]
	zs = []
	for i in n.layers
		push!(zs,i.W*x+i.b)
		push!(active,i(x))
		x = active[end]
	end
	return zs,active
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

function Base.:*(x::Float64,layer::Layer)
	layer.b .*= x
	layer.W .*= x
end

function Base.:*(x::Float64,model::Network)
	map(model.layers) do layer
		x*layer
	end
end

function Base.:-(layer::Layer,delta_layer::Layer)
	layer.b .-= delta_layer.b
	layer.W .-= delta_layer.W
end

function Base.:-(model::Network,delta_model::Network)
	map(model.layers,delta_model.layers) do m_l,d_l
		m_l-d_l
	end
end


function sgd(train_data,epochs,mini_batch_size,eta,model,delta_model,test_data=0)
	if test_data != 0
		n_test = length(test_data)
	end
	for epoch=1:epochs
		map(1:mini_batch_size:length(train_data)) do x
			rb = random_batch(train_data,mini_batch_size)
			updating_mini_batch(rb,eta,model,delta_model)
		end

		# Should be simplified by 1 function
		#push!(x_plot,sum(map(x->sum(cost(model(x[1])[2][3],x[2]))/length(test_data),test_data)))
		#push!(y_plot,epoch)
		if test_data != 0
			println("Epoch ",epoch,":",evaluate(test_data,model),"/",n_test)
			#display(x_plot[end])
		else
			println("Epoch ",epoch,"complited")
		end
	end
end


function random_batch(x::Vector,batch_size::Int)
	indx = sample(1:Base.size(x,1),batch_size,replace = false)
	return x[indx]
end


function updating_mini_batch(mini_batch,eta,model,delta_model)
	#updating 'w' and 'b' 
	#noble_b = map(x->zeros(size(x.b)),model.layers) 
	#noble_w = map(x->zeros(size(x.W)),model.layers)
	
	for i=1:length(mini_batch)
		zs, activations = model(mini_batch[i][1])
		cost_ = cost_derivative(activations[end],mini_batch[i][2])
		back(zs,activations,cost_,model,delta_model)
		((eta/length(mini_batch))*delta_model)
		model-delta_model
		

		# Old version
		#delta_b,delta_w = backprop(mini_batch[i][1],mini_batch[i][2],model)
		#all_W,all_b = map(l->l.W,model.layers),map(l->l.b,model.layers)

		#noble_w = noble_w+delta_w
		#noble_b = noble_b+delta_b
	
		#new_W = all_W-(eta/length(mini_batch)).*noble_w
		#new_b = all_b-(eta/length(mini_batch)).*noble_b
		#for i=1:length(model.layers) 
		#	model.layers[i].W .= new_W[i]
		#	model.layers[i].b .= new_b[i]
		#end
			
	end
end



function back(zs::Vector,activations::Vector,cost_::Vector,model::Network,delta_model::Network)
	delta = 0
	for i = length(model.layers):-1:1
		sp = sigma_prime.(zs[i])
		delta = (i == length(model.layers) ? cost_ : model.layers[i+1].W'*delta).*sp
		delta_model.layers[i].b .+= delta
		delta_model.layers[i].W .+= delta*activations[i]'
	end
end



function backprop(x::Vector,y::Int,model::Network)
	noble_b = map(x->zeros(size(x.b)),model.layers) 
	noble_w = map(x->zeros(size(x.W)),model.layers)
	zs,activations = model(x) 

	delta = cost_derivative(activations[end],y).*sigma_prime.(zs[end])

	noble_b[end] .= delta 		
	noble_w[end] .= delta*activations[end-1]'

	for i = length(model.layers)-1:-1:1
		#z = zs[i]
		sp = sigma_prime.(zs[i])
		layer = model.layers[i+1]
		W = layer.W
		delta = (W'*delta).*sp
		noble_b[i] .= delta
		noble_w[i] .= delta*activations[i]'
	end

	return noble_b,noble_w
end

function cost(out_a,y)
	return sum(cost_derivative(out_a,y).^2)	
end


function cost_derivative(output_active,y)
	# making number y into array of zeros where y-th elem is 1 
	new_y = zeros(Base.size(output_active))
	new_y[y+1] = 1
	return (output_active-new_y)#sum((output_active-new_y).^2)
end


function evaluate(test_data,model)
	test_result = map(x->(findmax(model(x[1])[2][3])[2]-1,x[2]),test_data)
	#True = 1, False = 0
	return sum(map(x->x[1]==x[2],test_result)) 
end


function zipping(sample,label::Array)
	a = []
	for i=1:length(label)
		push!(a,(reshape(sample[:,:,i],length(sample[:,:,i])),label[i]))	
	end
	return a
end

function graph!(x_data::Array,y_data::Array)
	gr()
	plot!(x_data,y_data,label="Evol of cost")
	
end
#Initiating NN

training_x,train_y = MNIST.traindata(Float32)
test_x,test_y = MNIST.testdata(Float32)
tr_x = MNIST.traintensor(Float64)

model = Network(
	[Layer(784,30),
	Layer(30,10)]
)
delta_model = Network(
	[Layer(zeros(30),zeros(30,784)),
	Layer(zeros(10),zeros(10,30))]
)

data = zipping(training_x,train_y)
tst = zipping(test_x,test_y)
sgd(data,30,10,0.10,model,delta_model,tst)

