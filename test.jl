using Plots
using Random
using LinearAlgebra
using MLDatasets
using StatsBase

x_plot = []
y_plot = []
#cost function, new one
#write down into LaTex SGD
#Validation loss
#Later 

struct Layer
	b::Vector
	W::Matrix
end

struct Network
	layers::Vector{Layer}
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


function sgd(train_data,epochs,mini_batch_size,eta,model,test_data=0)
	if test_data != 0
		n_test = length(test_data)
	end
	for epoch=1:epochs
		#mini_batches = train_data[shuffle(1:end)]
		map(mini_batch->updating_mini_batch(random_batch(train_data,mini_batch_size),eta,model),1:mini_batch_size:length(train_data))
		test_data = test_data[shuffle(1:end)]
		push!(x_plot,sum(map(x->sum(cost(model(x[1])[2][3],x[2]))/length(test_data),test_data)))
		push!(y_plot,epoch)
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

		noble_w = noble_w+delta_w
		noble_b = noble_b+delta_b
	
		new_W = map((_W,n_w)->_W-(eta).*n_w,all_W,noble_w)
		new_b = map((_b,n_b)->_b-(eta).*n_b,all_b,noble_b)
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

	delta = 2*cost_derivative(activations[end],y).*sigma_prime.(zs[end])

	noble_b[end] .= delta 		
	noble_w[end] .= delta.*activations[end-1]'
	
	for i = length(model.layers)-1:-1:1
		z = zs[i]
		sp = sigma_prime.(z)
		layer = model.layers[i+1]
		W,b = layer.W,layer.b
		delta = (W'*delta).*sp
		noble_b[i] .= delta
		noble_w[i] .= delta.*activations[i]'
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
	test_result = map(x->(findmax(model(x[1]))[2]-1,x[2]),test_data)
	#display(test_result)
	#True = 1, False = 0
	return cumsum(map(x->x[1]==x[2],test_result),dims=1)[end]#[length(map(x->x[1]==x[2],test_result))]
end


function zipping(sample,label::Array)
	a = []
	for i=1:length(label)
		push!(a,(reshape(sample[:,:,i],length(sample[:,:,i])),label[i]))	
	end
	return a
end


#Initiating NN

training_x,train_y = MNIST.traindata(Float32)
test_x,test_y = MNIST.testdata(Float32)
tr_x = MNIST.traintensor(Float64)
println("<============================>")



model = Network(
	[Layer(784,30),
	Layer(30,10)]
)

data = zipping(training_x,train_y)
tst = zipping(test_x,test_y)
display("====")
sgd(data[1:10000],30,10,0.01,model,tst[1:1000])

#Ploting evol of cost 
gr()
plot(y_plot,x_plot,label="evol of cost")
scatter!(y_plot,x_plot,label="evol of cost")
xlabel!("Number of epoch")
ylabel!("MSE")
png("C:/Users/alexs/Julia/test\\plot")
