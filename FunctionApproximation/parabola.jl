using Flux
using NeuralArithmetic
using Plots
using LinearAlgebra
using Random
Random.seed!(1)

f(x) = x^2
file = "SimplePol"

range = (-1,10)

X = Float32.(reshape(collect(-1.5f0:0.067f0:1.5f0), 1, :))

Y = f.(X)

data = [(X,Y) for _ in 1:80000]


hdim = 24
npu = Chain(NPU(1,hdim), NAU(hdim,1))
nmu = Chain(NAU(1,12),NMU(12,12),NAU(12,1))
dense = Chain(Dense(1,hdim,sigmoid), Dense(hdim,1))
nalu = Chain(NALU(1,hdim), NALU(hdim,1))

function makeplotFinal(model,name)
    p1 = Plots.plot(-3f0:0.013f0:3f0, x->model([x])[1],line=(3),label=(name))
    Plots.plot!(p1, -3f0:0.013f0:3, f, ylims=range,line=(:dash,2),label="Truth",legend=:top)
    Plots.scatter!(p1, vec(X), vec(Y),label="Data")
end

opt = ADAM(1e-4)
beta = 1e-4
sqnorm(x) = sum(abs,x)

function makeplot(model)	
    p1 = plot(2f0:0.013f0:2f0, x->model([x])[1],line=(3))
    plot!(p1, -2f0:0.013f0:2f0, f, ylims=range,line=(:dash,2),legend=:bottomright)
	scatter!(p1, vec(X), vec(Y))
end


cb = [Flux.throttle(()->(@info "NMU" loss(X,Y)),1),
      Flux.throttle(()->(display(makeplot(nmu))),1)]

loss(X,Y) = Flux.mse(nmu(X), Y)+beta*sum(sqnorm,Flux.params(nmu))
Flux.train!(loss, Flux.params(nmu), data, opt, cb=cb)


cb = [Flux.throttle(()->(@info "NPU" loss(X,Y)),1),
      Flux.throttle(()->(display(makeplot(npu))),1)]
loss(X,Y) = Flux.mse(npu(X), Y) +beta*sum(sqnorm,Flux.params(npu))
Flux.train!(loss, Flux.params(npu), data, opt, cb=cb)


error(1)
cb = [Flux.throttle(()->(@info "Dense" loss(X,Y)),1),
      Flux.throttle(()->(display(makeplot(dense))),1)]
loss(X,Y) = Flux.mse(dense(X), Y) +beta*sum(sqnorm,Flux.params(dense))
Flux.train!(loss, Flux.params(dense), data, opt, cb=cb)



cb = [Flux.throttle(()->(@info "NALU" loss(X,Y)),1),
      Flux.throttle(()->(display(makeplot(nalu))),1)]
loss(X,Y) = Flux.mse(nalu(X), Y) +beta*sum(sqnorm,Flux.params(nalu))
Flux.train!(loss, Flux.params(nalu), data, opt, cb=cb)


plt1 = makeplotFinal(npu,"NPU")
plt2 = makeplotFinal(dense,"Dense")
plt3 = makeplotFinal(nalu,"NALU")
plt4 = makeplotFinal(nmu,"NMU")
Plots.plot!(plt3,plt4,plt1,plt2,layout=(2,2))
savefig(string(file,"/ArchComparisonNALM's.pdf"))


function makePlotBigFinal(model,name,lines)
    Plots.plot!(-3.f0:0.013f0:3f0, x->model([x])[1],line=(3,lines),label=(name))
end

Plots.scatter(vec(X), vec(Y),label="Data")
Plots.plot!(-3f0:0.013f0:3, f, ylims=range,line=(2),label="Truth",legend=:top)
makePlotBigFinal(npu,"NPU",:dash)
makePlotBigFinal(dense,"Dense",:line)
makePlotBigFinal(nalu,"NALU",:dashdot)
makePlotBigFinal(nmu,"NMU",:dot)
savefig(string(file,"/TotalArchComparisonNALM's.pdf"))

using BSON: @save
@save string(file,"/npuModel.bson") npu
@save string(file,"/naluModel.bson") nalu
@save string(file,"/nmuModel.bson") nmu
@save string(file,"/denseModel.bson") dense

function modelsError(model,name)
	Xv = Float32.(reshape(collect(-3f0:0.067f0:3f0), 1, :))
	Yv = f.(Xv)
	println(name," training: ",Flux.mse(model(X),Y))
	println(name," validation: ",Flux.mse(model(Xv),Yv)-Flux.mse(model(X),Y))
end

modelsError(nalu,"NALU")
modelsError(npu,"NPU")
modelsError(nmu,"NMU")
modelsError(dense,"Dense")
