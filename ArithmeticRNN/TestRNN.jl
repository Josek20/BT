using Flux
using MLDatasets
using StatsBase

test_model = Chain(Flux.LSTM(6,10),Flux.LSTM(10,10),Dense(10,1))

max_sequence_size = 3
lrange = -2.
rrange = 2.
batch_length = 30
epochs = 10
training_length = 6000
validation_length = 6000
v_lrange = -4.
v_rrange = 4.
train_data,train_labels = MNIST.traindata(Float32)
validation_data,validation_labels = MNIST.testdata(Float32)

w,h,n_t = size(train_data)
_,_,n_v = size(validation_data)


train_data = reshape(train_data,w*h,n_t)
validation_data = reshape(validation_data,w*h,n_v)


function random_sequence(sequence_size,data,label,range)
	batch = []
	lrange,rrange = range
	lbls = [zeros(Float32,1,batch_length)]
	
	for i in 1:sequence_size
		x = Float32.(rand(Uniform(lrange,rrange),1,batch_length))

		operations = zeros(Float32,3,batch_length)
		operations[3,:] .+= ones(Float32,batch_length)
		operations[2,:] .+= rand((0,1),batch_length)
	
		operations1 = ones(Float32,3,batch_length)
		operations1 .= operations
		operations[3,:] .-= operations[2,:] 

		if i == 1
			operations = [zeros(Float32,1,batch_length);ones(Float32,1,batch_length);zeros(Float32,1,batch_length)]
		end

		append!(batch,[[operations;[ones(Float32,1,batch_length).*x;zeros(Float32,2,batch_length)]]])
	    
		if i == 1
			append!(lbls,[lbls[end] .+ reshape(batch[end][4,:],1,batch_length)])
		else
			a = operations1[3,:] .- operations1[2,:]
			c = [zeros(Float32,1,batch_length);reshape(operations1[2,:] .* lbls[end][1,:],1,batch_length); reshape((a .* lbls[end][1,:]) .+ operations1[2,:],1,batch_length)]
			cb = reshape((c[3,:] .* batch[end][4,:]) .+ c[2,:] ,1,batch_length)
			append!(lbls,[cb])
		end
	end
	(batch,lbls[2:end])
end


data = [random_sequence(rand(2:max_sequence_size),train_data,train_labels,(lrange,rrange)) for _ in 1:training_length]
v_data = [random_sequence(rand(2:max_sequence_size),validation_data,validation_labels,(v_lrange,v_rrange)) for _ in 1:validation_length]


function evaluation(x)
	output = test_model.(x)
	Flux.reset!(test_model)
	return output
end

loss(xs,ys) = sum(map(Flux.mse,evaluation(xs),ys))

pc = Flux.params(test_model)
opt = ADAM()
t_error = []
v_error = []

for i in 1:epochs
	local training_loss
	print("number of epoch: ",i,"\n")
	for d in data
		gs = gradient(pc) do
			training_loss = loss(d...)
			return training_loss
		end
		Flux.update!(opt,pc,gs)
	end
	append!(v_error,sum([loss(j...) for j in v_data])/validation_length), 
	append!(t_error,sum([loss(j...) for j in data])/training_length)
	display(t_error[end])
end

using BSON: @save
@save "LSTM_6to1_RandomOp_rnn_model.bson" test_model
