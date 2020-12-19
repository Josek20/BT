using Flux
using Flux:@epochs
using Flux.Data: DataLoader
using Flux:gradient,params
using Flux.Optimise: update!,Descent
using Distributions
using LinearAlgebra
using NeuralArithmetic
using Plots
using Random

Random.seed!(0)

#few functions: squer, some polynomial, logorithm 
#and their values on the plot
#how it works on different ranges, like shift the range,
#or make it biger smaller. How does renges afferc generalization perfomance

beta = 1e-4
lr = 1e-3
batchsize = 100
lrange = 0.01
rrange = 2
function task(x::Vector)
	#a,b = x[1],x[2]
	#[a+b,a-b,a*b,a/b]
	[x.^2 + 2*x;sqrt.(x);log.(x);abs.(x)]
end

#task(x::Matrix) = mapslices(task,x,dims=1)

function generate_data(bs::Int)
	a = Uniform(lrange,rrange)
	b = Uniform(0.01,rrange)
	#x = Float32.(rand(Product([a,b]),bs))
	x = Float32.(rand(Product([a]),bs))
	y = task(x)
	(x,y)
end


data = [generate_data(batchsize) for _ in 1:5000]

nac_model = Chain(NALU(1,12),NALU(12,4))
pc = Flux.params(nac_model)
cost(x,y) = Flux.mse(nac_model(x),y)
loss(x,y) = cost(x,y)
opt = ADAM(lr)
cb = Flux.throttle(()->(@info "training..." loss(data[1]...) cost(data[1]...)),1)
Flux.train!(cost,pc,data,opt,cb=cb)
tm = [data[1][2][1,:],nac_model(data[1][1])[1,:]]
scatter(data[1][1][1:batchsize],tm)
savefig("test_train.png")



