using Flux:@epochs
using Flux.Data: DataLoader
using Flux:gradient,params
using Flux.Optimise: update!,Descent
using MLDatasets

xtrain,ytrain = MNIST.traindata()
xtest,ytest = MNIST.testdata()

xtrain = Flux.flatten(xtrain)
xtest = Flux.flatten(xtest)
out_size = 10
eta = 0.1

function onehot(y,sz)
	tmp = zeros(sz)
	tmp[y+1] = 1
	return tmp
end

function accuracy(data,model)
	acc = 0
	for (x,y) in data
		acc+=sum(Flux.onecold(model(x)) .== Flux.onecold(y))*1/size(x,2)
	end
	acc/length(data)
end

ytrain = Flux.onehotbatch(ytrain,0:9)
#onehot.(ytrain,out_size*ones(Int64,size(ytrain)))
ytest = Flux.onehotbatch(ytest,0:9)
#onehot.(ytest,out_size*ones(Int64,size(ytest)))


train_data = DataLoader(xtrain,ytrain,batchsize=1000,shuffle=true)
test_data = DataLoader(xtest,ytest,batchsize=1000)

#lossfunc(x) = sum(W*x+b);

#grads = gradient(()->lossfunc(x),params([W,b]))


model = Chain(Dense(784,30,relu),Dense(30,10),softmax)

loss(x,y) = sum(Flux.crossentropy(model(x),y))

eta = 3e-4
opt = Descent(eta)
@epochs 10 Flux.train!(loss,params(m),train_data,opt)

@show accuracy(test_data,model)

#for i = 1:1000:size(xtrain)[2]
#	gs = gradient(params(model)) do	
#		l += loss.(xtrain[:,i:i+1000],ytrain[i:i+1000])
#	end
#	update!(opt,params(model),gs)
	#for p in params(model)
	#	update!(p,-eta*gs[p])
	#end
	#display("Ok2")
#end

