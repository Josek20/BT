using Flux
using Plots
using MLDatasets
using StatsBase

#=
	my RNN cell implementation
=#


struct RNNcell{F,network,A,V,S}
  s::F
  Wi::network
  Wh::A
  b::V
  state0::S
end

RNNcell(in::Integer, out::Integer,network, s=tanh, init=Flux.glorot_uniform, initb=zeros, init_state=zeros) = 
  RNNcell(s, network, init(out, out), initb(out), init_state(out,1))

function (m::RNNcell)(h, x) 
  s, net, Wh, b = m.s, m.Wi, m.Wh, m.b
  h = s.(findmax(net(x))[2]-1 .+ Wh*h .+ b)
  #sz = size(x)
  return h, h #reshape(h, :, sz[2:end]...)
end

Flux.@functor RNNcell
network = Chain(Dense(784,30,relu),Dense(30,10),softmax)
RNN(a...; ka...) = Recur(RNNcell(a...; ka...))
Recur(m::RNNcell) = Flux.Recur(m, m.state0)
rnn_model = RNN(784,1,network)

#=
	End of RNN implementation
=#

max_batch_size = 10

epochs = 10

train_data,train_labels = MNIST.traindata(Float32)
validation_data,validation_labels = MNIST.testdata(Float32)

w,h,n_t = size(train_data)
_,_,n_v = size(validation_data)

train_data = reshape(train_data,w*h,n_t)
validation_data = reshape(validation_data,w*h,n_v)

#t_data = zip(train_data,train_labels)
#v_data = zip(validation_data,validation_labels)

#data_x = [rand(1:100,rand(2:10)) for _ in 1:sample_size]
#data_y = map(x->sum(x),data_x)
#data = zip(data_x,data_y)

function random_batch(sample_size)
	indx = sample(1:n_t,sample_size)
	#lbs = [i for i in train_labels[indx]]
	lbls = [0]
	for i in train_labels[indx]
		append!(lbls,lbls[end]+i)
	end
	(train_data[:,indx],lbls[2:end])
	#sum(train_labels[indx]))
end

function random_v_batch(sample_size)
	indx = sample(1:n_v,sample_size)
	lbls = [0]
	for i in validation_labels[indx]
		append!(lbls,lbls[end]+i)
	end
	(validation_data[:,indx],lbls[2:end])
end

data = [random_batch(rand(2:max_batch_size)) for _ in 1:6000]
v_data = [random_v_batch(rand(2:max_batch_size)) for _ in 1:1000]


function evaluation(x)
	Flux.reset!(rnn_model)
	output = 0
	for i in 1:size(x)[2]
		output = rnn_model(x[:,i])
	end
	return output
end

loss(x,y) = Flux.mae(evaluation(x),y[end])

pc = Flux.params(rnn_model)
opt = ADAM()
t_error = []
v_error = []
evalcb() = @show(
	append!(v_error,sum([loss(j...) for j in v_data])/1000), 
	append!(t_error,sum([loss(j...) for j in data])/6000)
	)
#error("ok")
Flux.@epochs epochs Flux.train!(loss,pc,data,opt,cb=Flux.throttle(evalcb,5))

#k_error = []
#for j in v_data
#	out = loss(j...)
#	append!(k_error,out)
#end

