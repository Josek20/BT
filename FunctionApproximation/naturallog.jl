using Flux
using NeuralArithmetic
using Plots
using LinearAlgebra
using Random
Random.seed!(1)

f(x) = log(x)
file = "NaturalLog"

range = (-5,3)

X = Float32.(reshape(collect(0.01f0:0.067f0:1.5f0), 1, :))

Y = f.(X)
data = [(X,Y) for _ in 1:80000]


hdim = 24
npu = Chain(NPU(1,hdim), NAU(hdim,1))
nmu = Chain(NAU(1,12),NMU(12,12),NAU(12,1))
dense = Chain(Dense(1,hdim,sigmoid), Dense(hdim,1))
nalu = Chain(NALU(1,hdim), NALU(hdim,1))

function makeplotFinall(model,name)
    p1 = Plots.plot(0.01f0:0.013f0:3f0, x->model([x])[1],line=(3),label=(name))
    Plots.plot!(p1, 0.01f0:0.013f0:3, f, ylims=range,line=(:dash,2),label="Truth",legend=:bottomright)
    Plots.scatter!(p1, vec(X), vec(Y),label="Data")
end

opt = ADAM(1e-4)
beta = 0
sqnorm(x) = sum(abs,x)

function makeplot(model)	
    p1 = plot(0.01f0:0.013f0:2f0, x->model([x])[1],line=(3))
    plot!(p1, 0.01f0:0.013f0:2f0, f, ylims=range,line=(:dash,2),legend=:bottomright)
	scatter!(p1, vec(X), vec(Y))
end


cb = [Flux.throttle(()->(@info "NMU" loss(X,Y)),1),
      Flux.throttle(()->(display(makeplot(nmu))),1)]

loss(X,Y) = Flux.mse(nmu(X), Y)+beta*sum(sqnorm,Flux.params(nmu))
Flux.train!(loss, Flux.params(nmu), data, opt, cb=cb)


cb = [Flux.throttle(()->(@info "NPU" loss(X,Y)),1),
      Flux.throttle(()->(display(makeplot(npu))),1)]
loss(X,Y) = Flux.mse(npu(X), Y)+beta*sum(sqnorm,Flux.params(npu))
Flux.train!(loss, Flux.params(npu), data, opt, cb=cb)


cb = [Flux.throttle(()->(@info "Dense" loss(X,Y)),1),
      Flux.throttle(()->(display(makeplot(dense))),1)]
loss(X,Y) = Flux.mse(dense(X), Y)+beta*sum(sqnorm,Flux.params(dense))
Flux.train!(loss, Flux.params(dense), data, opt, cb=cb)



cb = [Flux.throttle(()->(@info "NALU" loss(X,Y)),1),
      Flux.throttle(()->(display(makeplot(nalu))),1)]
loss(X,Y) = Flux.mse(nalu(X), Y)+beta*sum(sqnorm,Flux.params(nalu))
Flux.train!(loss, Flux.params(nalu), data, opt, cb=cb)


plt1 = makeplotFinall(npu,"NPU")
plt2 = makeplotFinall(dense,"Dense")
plt3 = makeplotFinall(nalu,"NALU")
plt4 = makeplotFinall(nmu,"NMU")
Plots.plot!(plt3,plt4,plt1,plt2,layout=(2,2))
savefig(string(file,"/ArchComparisonNALM's.pdf"))


function makePlotBigFinall(model,name,lines)
    Plots.plot!(0.01f0:0.013f0:3f0, x->model([x])[1],line=(3,lines),label=(name))
end

Plots.scatter(vec(X), vec(Y),label="Data")
Plots.plot!(0.01f0:0.013f0:3, f, ylims=range,line=(2),label="Truth",legend=:bottomright)
makePlotBigFinall(npu,"NPU",:dash)
makePlotBigFinall(dense,"Dense",:line)
makePlotBigFinall(nalu,"NALU",:dashdot)
makePlotBigFinall(nmu,"NMU",:dot)
savefig(string(file,"/TotalArchComparisonNALM's.pdf"))

using BSON: @save
@save string(file,"/npuModel.bson") npu
@save string(file,"/naluModel.bson") nalu
@save string(file,"/nmuModel.bson") nmu
@save string(file,"/denseModel.bson") dense

function modelsError(model,name)
	Xv = Float32.(reshape(collect(0.01f0:0.067f0:3f0), 1, :))
	Yv = f.(Xv)
	println(name," training: ",Flux.mse(model(X),Y))
	println(name," validation: ",Flux.mse(model(Xv),Yv)-Flux.mse(model(X),Y))
end

modelsError(nalu,"NALU")
modelsError(npu,"NPU")
modelsError(nmu,"NMU")
modelsError(dense,"Dense")
