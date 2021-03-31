using Flux
using Plots
using MLDatasets
using StatsBase
using NeuralArithmetic

#=
	Training settings
=#
max_sequence_size = 3
batch_length = 30
epochs = 100
training_length = 6000
validation_length = 1000

#=
	my RNN cell implementation
=#


struct DenseRNNcell{F,N,AN,V,S}
  s::F
  Wi::N
  Wh::AN
  b::V
  state0::S
end

DenseRNNcell(in::Integer, out::Integer,net,arith_net, s=tanh, init=Flux.glorot_uniform, initb=zeros, init_state=zeros) = 
  DenseRNNcell(s, net, arith_net, initb(Float32,1,batch_length), init_state(Float32,1,batch_length))

function (m::DenseRNNcell)(h, x)
  s, net, a_net, b = m.s, m.Wi, m.Wh, m.b
  tmp = net(x)
  h = a_net([tmp;h]) .+ b 
  sz = size(x)
  return h, reshape(h, :, sz[2:end]...)
end



Flux.@functor DenseRNNcell
# Todo:extra layer
network = Chain(Dense(784,120,relu),Dense(120,30,relu),Dense(30,1),softmax)
arithmetic_network = Chain(NPU(2,30),NPU(30,1))

RNN(a...;k...) = Recur(DenseRNNcell(a...;k...))
Recur(m::DenseRNNcell) = Flux.Recur(m,m.state0)
rnn_model = RNN(784,1,network,arithmetic_network) 

#=
	End of RNN implementation
=#



train_data,train_labels = MNIST.traindata(Float32)
validation_data,validation_labels = MNIST.testdata(Float32)

w,h,n_t = size(train_data)
_,_,n_v = size(validation_data)


train_data = reshape(train_data,w*h,n_t)
validation_data = reshape(validation_data,w*h,n_v)


function random_sequence(sequence_size,data,label)
	batch = []
	lbls = [[0]]
	for i in 1:sequence_size
		indx = sample(1:size(data)[2],batch_length)
		append!(batch,[data[:,indx]])
		append!(lbls,[lbls[end].+label[indx]])
	end
	(batch,lbls[2:end])
end


data = [random_sequence(rand(2:max_sequence_size),train_data,train_labels) for _ in 1:training_length]
v_data = [random_sequence(rand(2:max_sequence_size),validation_data,validation_labels) for _ in 1:validation_length]


function evaluation(x)
	output = rnn_model.(x)
	Flux.reset!(rnn_model)
	return output
end

#loss(x,y) = Float64(abs.(sum(evaluation(x) - y))[1])
loss(x,y) = Flux.mae(evaluation(x)[end],reshape(y[end],1,batch_length))

#error("ok")


pc = Flux.params(rnn_model)
opt = ADAM()
t_error = []
v_error = []
ev() = @show(loss(v_data[50]...))
Flux.@epochs epochs Flux.train!(loss,pc,data,opt,cb = Flux.throttle(ev,10))
v_acc = sum([loss(j...) for j in v_data])/validation_length
t_acc = sum([loss(j...) for j in data])/training_length

display(v_acc)
display(t_acc)

#=
for i in 1:epochs
	local training_loss
	print("number of epoch: ",i,"\n")
	counter = 0
	for d in data
		gs = gradient(pc) do
			training_loss = loss(d...)
			return training_loss
		end
		counter += 1
		#display(counter)
		Flux.update!(opt,pc,gs)
	end
	append!(v_error,sum([loss(j...) for j in v_data])/validation_length), 
	append!(t_error,sum([loss(j...) for j in data])/training_length)
	display(t_error[end])
end
=#

# Todo: save models after training 
# give some appropriate names,
# same with experiments

#using BSON:@save

#save "rnn_model.bson" dense_model

