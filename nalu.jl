using Flux:@epochs
using Flux.Data: DataLoader
using Flux:gradient,params
using Flux.Optimise: update!,Descent
using MLDatasets

xtrain,ytrain = MLDatasets.MNIST.traindata(Float32)
xtest,ytest = MLDatasets.MNIST.testdata(Float32)

xtrain = Flux.flatten(xtrain)
xtest = Flux.flatten(xtest)
eta = 3e-4

function onehot(y,sz)
	tmp = zeros(sz)
	tmp[y+1] = 1
	return tmp
end

function accuracy(data,model)
	acc = 0
	for (x,y) in data
		acc+=sum(Flux.onecold(cpu(model(x))) .== Flux.onecold(cpu(y)))*1/size(x,2)
	end
	acc/length(data)
end

ytrain = Flux.onehotbatch(ytrain,0:9)
ytest = Flux.onehotbatch(ytest,0:9)


train_data = DataLoader(xtrain,ytrain,batchsize=1000,shuffle=true)
test_data = DataLoader(xtest,ytest,batchsize=1000)

#lossfunc(x) = sum(W*x+b);

#grads = gradient(()->lossfunc(x),params([W,b]))


model = Chain(Dense(784,32,relu),Dense(32,10))

loss(x,y) = sum(Flux.logitcrossentropy(model(x),y))

opt = ADAM(eta)
@epochs 10 Flux.train!(loss,params(model),train_data,opt)

@show accuracy(train_data,model)

@show accuracy(test_data,model)
