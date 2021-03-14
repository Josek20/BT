using Flux
using Plots
using MLDatasets
using StatsBase

#=
	my RNN cell implementation
=#

#=
struct RNNcell{func,dense_,weights,biases,state}
	sgm::func
	Wi::dense_
	Wh::weights
	b::biases
	st::state
end


RNNcell(in,out,network) = RNNcell(tanh,network,Flux.glorot_uniform(out,out),zeros(out),zeros(out,1)) 
#RNNcell(in,out) = RNNcell(tanh,Flux.glorot_uniform(out,in),Flux.glorot_uniform(out,out),zeros(out),zeros(out,1)) 

function (model::RNNcell{func,dense_,weights,biases})(h,x) where {func,dense_,weights,biases}
	funct,dense,Wh,b,st = model.sgm, model.Wi, model.Wh, model.b, model.st
	h = funct.(Wh*h .+ findmax(dense(x),dims=1)[1] .+ b)
	return h,h	
end

#@functor RNNcell

network = Chain(Dense(784,30,relu),Dense(30,10),softmax) 

#m = RNNcell(784,1,network)
#new_model = Flux.RNN(m,m.st)

Flux.RNN(a...; ka...) = Flux.Recur(RNNcell(a...; ka...))
Flux.Recur(m::RNNcell) = Flux.Recur(m,m.st)
=#


struct RNNCell{F,network,A,V,S}
  s::F
  Wi::network
  Wh::A
  b::V
  state0::S
end

RNNCell(in::Integer, out::Integer,network, s=tanh, init=Flux.glorot_uniform, initb=zeros, init_state=zeros) = 
  RNNCell(s, network, init(out, out), initb(out), init_state(out,1))

function (m::RNNCell{F,network,A,V,<:AbstractMatrix{T}})(h, x) where {F,network,A,V,T}
  s, net, Wh, b = m.s, m.Wi, m.Wh, m.b
  print("ok1")
  a = map(x->x[1]-1,findmax(net(x),dims=1)[2])
  print("ok2")
  h = s.(a .+ Wh*h .+ b)
  print("ok3")
  #sz = size(x)
  return h, h #reshape(h, :, sz[2:end]...)
end

Flux.@functor RNNCell
network = Chain(Dense(784,30,relu),Dense(30,10),softmax)
RNN(a...; ka...) = Recur(RNNCell(a...; ka...))
Recur(m::RNNCell) = Recur(m, m.state0)
rnn_model = Flux.RNN(784,1,network)

"""
	End of RNN implementation
"""

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
	output = rnn_model(x)[end]
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

k_error = []
for j in v_data
	out = loss(j...)
	append!(k_error,out)
end

