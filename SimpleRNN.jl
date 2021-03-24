using Flux
using Plots
using MLDatasets
using StatsBase

#=
	my RNN cell implementation
=#


struct DenseRNNcell{F,N,A,V,S}
  s::F
  Wi::N
  Wh::A
  b::V
  state0::S
end

DenseRNNcell(in::Integer, out::Integer,net, s=tanh, init=Flux.glorot_uniform, initb=zeros, init_state=zeros) = 
  DenseRNNcell(s, net, init(out, out), initb(out), init_state(out,1))

function (m::DenseRNNcell)(h, x)
  s, net, Wh, b = m.s, m.Wi, m.Wh, m.b
  tmp = net(x)
  h = tmp.+ Wh*h .+ b
  sz = size(x)
  return h, reshape(h, :, sz[2:end]...)
end


#=
function DenseRNNcell(network)
  out = 1#Flux.outdims(network,784)
  init = Flux.glorot_uniform
  initb = zeros
  init_state = zeros
  DenseRNNcell(tanh, network, init(out, out), initb(out), init_state(out,1))
end
=#


Flux.@functor DenseRNNcell
# Todo:extra layer
network = Chain(Dense(784,30,relu),Dense(10,1))

RNN(a...;k...) = Recur(DenseRNNcell(a...;k...))
Recur(m::DenseRNNcell) = Flux.Recur(m,m.state0)
rnn_model = RNN(784,1,network) 
#=
	End of RNN implementation
=#

max_sequence_size = 3
batch_length = 1
epochs = 30

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


data = [random_sequence(rand(2:max_sequence_size),train_data,train_labels) for _ in 1:10000]
v_data = [random_sequence(rand(2:max_sequence_size),validation_data,validation_labels) for _ in 1:1000]


function evaluation(x)
	output = rnn_model.(x)
	Flux.reset!(rnn_model)
	return output
end

loss(x,y) = abs.(sum(evaluation(x) - y))[1]
#loss(x,y) = Flux.mae(evaluation(x)[end],y[end])
#loss(x,y) = (sum((evaluation(x) - y).^2)/2)[1]

pc = Flux.params(rnn_model)
opt = ADAM()
t_error = []
v_error = []

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
		Flux.update!(opt,pc,gs)
	end
	append!(v_error,sum([loss(j...) for j in v_data])/1000), 
	append!(t_error,sum([loss(j...) for j in data])/6000)
	display(t_error[end])
end

# Todo: save models after training 
# give some appropriate names,
# same with experiments

#using BSON:@save

#save "rnn_model.bson" dense_model

