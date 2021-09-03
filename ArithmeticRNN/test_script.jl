using BSON
using Flux
using MLDatasets
using StatsBase
using Plots
using NeuralArithmetic

mutable struct MyRecur{T,S}
  cell::T
  state::S
end

function (m::MyRecur)(x)
  op,num = x
  m.state, y = m.cell(m.state, op,num)
  return y
end

Flux.@functor MyRecur
Flux.trainable(a::MyRecur) = (a.cell,)
Flux.reset!(m::MyRecur) = (m.state = m.cell.state0)

struct DenseRNNcell{F,ON,AN,V,S}
  s::F
  Wi::ON
  Wh::AN
  b::V
  state0::S
end

DenseRNNcell(out::Int,opp_net,arith_net, s=tanh, init=Flux.glorot_uniform, initb=zeros, init_state=zeros) = 
  DenseRNNcell(s,opp_net, arith_net, initb(Float32,1,1), init_state(Float32,3,1))

function (m::DenseRNNcell)(h, op, x)
  s,op_net, a_net, b = m.s,m.Wi, m.Wh, m.b
  h = a_net([x;op_net([op;h .* ones(Float32,size(x))])]) .+ b 
  sz = size(x)
  return h, reshape(h, :, sz[2:end]...)
end

Flux.@functor DenseRNNcell
operation_network = Chain(Dense(6,10),Dense(10,3)) 
arithmetic_network_nmu = Chain(NAU(6,6),NMU(6,3))
arithmetic_network_npu = Chain(NAU(6,6),NPU(6,3))
arithmetic_network_nalu = Chain(NALU(6,6),NALU(6,3))

MyRecur(m::DenseRNNcell) = MyRecur(m,m.state0)

nmu_model = MyRecur(DenseRNNcell(1,operation_network,arithmetic_network_nmu)) 
npu_model = MyRecur(DenseRNNcell(1,operation_network,arithmetic_network_npu)) 
nalu_model = MyRecur(DenseRNNcell(1,operation_network,arithmetic_network_nalu)) 

model = Chain(npu_model,NAU(3,1))
model_npu = Chain(npu_model,NAU(3,1))
BSON.@load "NPUNAU_RandomOp_rnn_model.bson" model
model_npu = model

model = Chain(nmu_model,NAU(3,1))
model_nmu = Chain(nmu_model,NAU(3,1))
BSON.@load "NMUNAU_RandomOp_rnn_model.bson" model
model_nmu = model

model = Chain(nalu_model,NAU(3,1))
model_nalu = Chain(nalu_model,NAU(3,1))
BSON.@load "NALU_RandomOp_rnn_model.bson" model
model_npu = model

test_model = Chain(Flux.LSTM(6,30),Flux.LSTM(30,10),Dense(10,3))
BSON.@load "LSTM_6to1_RandomOp_rnn_model.bson" test_model

display("Loading models complete")


train_data,train_labels = MNIST.traindata(Float32)
validation_data,validation_labels = MNIST.testdata(Float32)

w,h,n_t = size(train_data)
_,_,n_v = size(validation_data)

train_data = reshape(train_data,w*h,n_t)
validation_data = reshape(validation_data,w*h,n_v)

max_sequence_size = 10
batch_length = 1000

function random_sequence(sequence_size,data,label,range)
	batch = []
	lrange,rrange = range
	lbls = [zeros(Float32,1,batch_length)]
	
	for i in 1:sequence_size
		#indx = sample(1:size(data)[2],batch_length)
		x = Float32.(rand(Uniform(lrange,rrange),1,batch_length))

		operations = zeros(Float32,3,batch_length)
		operations[3,:] .+= ones(Float32,batch_length)
		operations[2,:] .+= rand((0,1),batch_length)
	
		operations1 = ones(Float32,3,batch_length)
		operations1 .= operations
		operations[3,:] .-= operations[2,:] 

		#operations1 = [zeros(Float32,2,batch_length);ones(Float32,2,batch_length)]
		#operations = [zeros(Float32,1,batch_length);ones(Float32,1,batch_length);zeros(Float32,1,batch_length)]
		#operations = [zeros(Float32,2,batch_length);ones(Float32,1,batch_length)]

		if i == 1
			operations = [zeros(Float32,1,batch_length);ones(Float32,1,batch_length);zeros(Float32,1,batch_length)]
		end
		#operations1 .= operations

		append!(batch,[(operations,[ones(Float32,1,batch_length).*x;zeros(Float32,2,batch_length)])])
		#append!(batch,[(operations,[ones(Float32,1,batch_length).*reshape(label[indx],1,batch_length);zeros(Float32,2,batch_length)])])
		
	    
		if i == 1
			append!(lbls,[lbls[end] .+ reshape(batch[end][2][1,:],1,batch_length)])
		else
			a = operations1[3,:] .- operations1[2,:]
			c = [zeros(Float32,1,batch_length);reshape(operations1[2,:] .* lbls[end][1,:],1,batch_length); reshape((a .* lbls[end][1,:]) .+ operations1[2,:],1,batch_length)]
			cb = reshape((c[3,:] .* batch[end][2][1,:]) .+ c[2,:] ,1,batch_length)
			append!(lbls,[cb])
		end
	end
	(batch,lbls[2:end])
end


function evaluation(x,model)
	output = model.(x)
	Flux.reset!(model)
	return output
end


loss(xs,ys,model) = sum(map(Flux.mse,evaluation(xs,model),ys))

extrapolation_data = [random_sequence(3,validation_data,validation_labels,(-i,i)) for i in 2:1:max_sequence_size]

extrapolation_sequance = [random_sequence(i,validation_data,validation_labels,(-2.,2.)) for i in 2:1:max_sequence_size]

display("Generating data complete")
arithmetic_error_npu = []
arithmetic_error_nmu = []
arithmetic_error_nalu = []
classic_error = []

for j in extrapolation_data
	append!(arithmetic_error_npu,loss(j...,model_npu))
	append!(arithmetic_error_nmu,loss(j...,model_nmu))
	append!(arithmetic_error_nalu,loss(j...,model_nalu))
	d = map(x->[x[1];x[2]],j[1])
	append!(classic_error,loss(d,j[2],test_model))
end
tmp = [classic_error,arithmetic_error_nalu,arithmetic_error_npu,arithmetic_error_nmu]

styles = filter((s->begin
                s in Plots.supported_styles()
            end), [:solid, :dash, :dot, :dashdot, :dashdotdot])

styles = reshape(styles, 1, length(styles))
plot(2:max_sequence_size,tmp,xlabel="Scalar range [-x,x]",ylabel="Loss",label=["LSTM" "NALU" "NPU" "NMU"],yaxis=:log,legend=:bottomright,line=([4 4 4 4],styles))
savefig("DataExtrapolaitonNALMSvsLSTM")
display("Data extrapolation complete")

arithmetic_error_npu = []
arithmetic_error_nmu = []
arithmetic_error_nalu = []
classic_error = []

for j in extrapolation_sequance
	append!(arithmetic_error_npu,loss(j...,model_npu))
	append!(arithmetic_error_nmu,loss(j...,model_nmu))
	append!(arithmetic_error_nalu,loss(j...,model_nalu))
	d = map(x->[x[1];x[2]],j[1])
	append!(classic_error,loss(d,j[2],test_model))
end
tmp = [classic_error,arithmetic_error_nalu,arithmetic_error_npu,arithmetic_error_nmu]

plot(2:max_sequence_size,tmp,xlabel="Sequance length",ylabel="Loss",label=["LSTM" "NALU" "NPU" "NMU"],yaxis=:log,legend=:bottomright,line=([4 4 4 4],styles))
savefig("SequanceExtrapolaitonNALMSvsLSTM")
display("Sequance extrapolation complete")
