import torch
from torch import *
from torch import optim
from torch import empty
from dlc_practical_prologue import *
from torch import nn

import time #used for benchmarking

def generate_digit_sets(train_input, train_classes, test_input, test_classes):
    """
    Utility function to obtain single digit sets
    
    Args:
        train_input (torch.Tensor of size (N,2,height,width)): N training pairs
        train_classes (torch.Tensor of size (N,2)): N training classes
        test_input (torch.Tensor of size (N,2,height,width)): N test pairs
        test_classes (torch.Tensor of size (N,2)) : N test classes

    Returns:
        digit_train_input(torch.Tensor of size (N*2,1,height,width)): 2*N training single digits
        digit_train_target(torch.Tensor of size (N*2,number of classes)): 2*N training classes (one-hot encoded)
        digit_test_input(torch.Tensor of size (N*2,1,height,width)): 2*N test single digits
        digit_test_target(torch.Tensor of size (N*2,number of classes)): 2*N test classes (one-hot encoded)
        
    """
    num_samples = train_input.size(0) * 2
    digit_train_input = train_input.view(-1, 1, 14, 14)
    digit_test_input = test_input.view(-1, 1, 14, 14)
    
    digit_train_classes = train_classes.view(-1)
    digit_test_classes = test_classes.view(-1)
    digit_train_target = Tensor(num_samples, 10).zero_()
    digit_test_target = Tensor(num_samples, 10).zero_()
    for i in range(10):
        digit_train_target[:,i] = (digit_train_classes == i) * 1.0
        digit_test_target[:,i] = (digit_test_classes == i) * 1.0
    
    return digit_train_input, digit_train_target, digit_test_input, digit_test_target

def total_number_of_params(model):
    """
    Return the total number of trainable parameters of a model

    Args:
        model (torch.nn): pytorch neural net model

    Returns:
        int: number of trainable parameters of a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def prediction_from_digit(digit_output):
    """
    Return the predicted class of a comparaison between 2 digits.
    Requires a tensor whose main dimension is divisible by 2!

    Args:
        digit_output (torch.Tensor): output of a digit regression of size (2*N, 10)

    Returns:
        torch.Tensor of size(N, 1): class of the comparaison of 2 consecutive digits
            a row is set at 1 if the first digit is smaller than the second, 0 otherwise
    """ 
    _, predicted_digit_classes = digit_output.max(1)
    predicted_digit_pair = predicted_digit_classes.view(-1, 2)
    predicted_classes = (predicted_digit_pair[:,0] - predicted_digit_pair[:,1] <= 0) * 1
    return predicted_classes


def prediction_from_comparaison(comp_output):
    """
    Return the predicted class of a comparaison from a regression.

    Args:
        comp_output (torch.Tensor): output of a digit classifier of size (N, 1)

    Returns:
        torch.Tensor of size(N, 1): a row is set at 1 if the regression output 
            is bigger than 0.5, 0 otherwise
    """
    return (comp_output >= 0.5) * 1.0
   
    
def compute_nb_errors(predicted, target):
    """
    Compute the number of errors between a prediction and a target
    
    Args:
        predicted (torch.Tensor of size (N, D)): result from a model predictions
        target (torch.Tensor of size (N, D)): target classes
        
    Returns:
        float: error rate
        int: total number of errors
        int: number of samples
    """
    error = (predicted != target.view(predicted.shape)) * 1.0
    
    return error.mean().item(), int(error.sum().item()), error.shape[0]


def train_model(model, train_input, train_target, aux_train_target=None, mini_batch_size=100, lr=0.1, iterations=25, aux_loss_weight=0.5):
    """
    Train a pytorch model with provided input and output.
    Can also add an auxiliary target to avoid vanishing gradients
    and get better accuracy.

    Args:
        model (torch.nn): pytorch neural net model
        train_input (torch.Tensor): training set
        train_target (torch.Tensor): target set
        aux_train_target (torch.Tensor): auxiliary target, None by default
        aux_loss_weight (torch.Float) : auxiliary weight, 0.5 by default
        mini_batch_size (int): batch size for data loading/training, 100 by default
        lr (float): learning rate, 0.1 by default
        iterations(int) : number of epochs, 25 by default
    
    Returns:
        training_loss: Training error evolution during training
        training_error: Training loss evolution during training
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    training_loss = torch.empty((iterations,1)) #Training loss over epochs
    training_error = torch.empty((iterations,1)) #Training error over epochs
    
    for e in range(iterations):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            if aux_train_target is None:
                output = model(train_input.narrow(0, b, mini_batch_size))
                target = train_target.narrow(0, b, mini_batch_size)
                loss = criterion(output, target)
                error, _, _ = compute_nb_errors(prediction_from_comparaison(output),target)
                
            # In the case we use also an auxiliary loss
            else:
                output, output_aux = model(train_input.narrow(0, b, mini_batch_size))
                target = train_target.narrow(0, b, mini_batch_size)
                aux_size_factor = int(output_aux.shape[0] / output.shape[0])
                target_aux = aux_train_target.narrow(0, aux_size_factor * b, aux_size_factor * mini_batch_size)
                # Here, heavily penalizing auxiliary loss can result to better classification
                loss = criterion(output, target) + aux_loss_weight * criterion(output_aux, target_aux)
                error, _, _ = compute_nb_errors(prediction_from_comparaison(output),target)
            
            loss.backward()
            sum_loss = sum_loss + loss.item()
            optimizer.step()
            
        
        training_loss[e] = sum_loss
        training_error[e] = error
        print(e, sum_loss, end = '\r')

    return training_loss, training_error


def evaluate_model(model_generator, lr = 5e-4, is_digit_model=False, use_aux_target=False, rounds=25, iterations = 25, aux_loss_weight=None):
    """
    Evaluate a pytorch model with provided model.
    Estimate test performance of model over several rounds.

    Args:
        model_generator (func): function returning pytorch neural net
        is_digit_model (boolean): True if model trains on single digits, False if model trains on pairs of digits
        use_aux_target (boolean): True if model trains with an auxiliary loss, False otherwise
        aux_loss_weight (boolean): weight of auxiliary loss, None by default
        rounds (int): number of rounds, 25 by default
        lr (float): learning rate, 5e-4 by default
    
    Returns:
        float: test error mean estimate
        float: test error standard deviation estimate
        float: training error mean estimate
        torch.Tensor : loss estimate over trainings
    """
    test_error = torch.empty((rounds,1))
    train_error = torch.empty((rounds,1))
    training_errors = torch.empty((rounds, iterations))
    for k in range(rounds):
        model = model_generator()
        
        num_samples = 1000
        
        #Generate data
        comp_train_input, comp_train_target, comp_train_classes, comp_test_input, comp_test_target, comp_test_classes = generate_pair_sets(num_samples)
        digit_train_input, digit_train_target, digit_test_input, digit_test_target = generate_digit_sets(comp_train_input, comp_train_classes, comp_test_input, comp_test_classes)
        
        #Normalize data
        mu, std = comp_train_input.mean(), comp_train_input.std();
        comp_train_input.sub_(mu).div_(std);
        comp_test_input.sub_(mu).div_(std);
        
        #Choose input in function of model (is_digit/auxiliary loss/normal)
        aux_train_target = digit_train_target if use_aux_target else None       
        model_train_input = digit_train_input if is_digit_model else comp_train_input
        model_train_target = digit_train_target if is_digit_model else comp_train_target.float().view(-1,1)
        model_test_input = digit_test_input if is_digit_model else comp_test_input
        
        #Train model
        _, err = train_model(model, model_train_input, model_train_target, mini_batch_size=50, lr=lr, aux_train_target=aux_train_target, aux_loss_weight=aux_loss_weight, iterations = iterations)
        training_errors[k] = err.t()
        
        #Test model on test data
        output = model(model_test_input)
        if use_aux_target:
            output = output[0]
        #Predict from output and compute error
        predicted = prediction_from_digit(output) if is_digit_model else prediction_from_comparaison(output)        
        err,_,_ = compute_nb_errors(predicted, comp_test_target)
        test_error[k] = err
        
        #Test model on train data
        output = model(model_train_input)
        if use_aux_target:
            output = output[0]
        #Predict from output and compute error
        predicted = prediction_from_digit(output) if is_digit_model else prediction_from_comparaison(output)        
        err,_,_ = compute_nb_errors(predicted, comp_train_target)
        train_error[k] = err
        
        
    return (torch.mean(test_error).item(), torch.std(test_error).item(), torch.mean(train_error).item(), torch.std(train_error).item(), torch.mean(training_errors, axis = 0))

def split_dataset(train_input, train_target, k, fold):
    """
    Splits a training set into a training and validation set, for cross-validation.
    
    Args:
        train_input(torch.Tensor): training data
        train_target(torch.Tensor): training label
        fold (int): number of folds
        k (int): fold number
        
    Returns:
        val_input_set: validation input set
        val_target_set: validation label set
        train_input_set: training input set
        train_target_set: training label set
    """
    n = train_input.size(0)
    size = int(np.floor(n/fold))
    
    #Validation set
    val_input_set = train_input.narrow(0, k*size, size )
    val_target_set = train_target.narrow(0,k*size,size)
    
    #Training set
    if k == 0:
        train_input_set = train_input.narrow(0, size, n - size)
        train_target_set = train_target.narrow(0, size, n - size)

    elif size*k == n - size:
        train_input_set = train_input.narrow(0, 0, n - size )
        train_target_set = train_target.narrow(0, 0, n - size )
    else:
        train_input_set = torch.cat((train_input.narrow(0, 0, k*size),
                                        train_input.narrow(0, (k+1)*size, n -(k+1)*size )), 0)
        train_target_set = torch.cat((train_target.narrow(0, 0, k*size),
                                        train_target.narrow(0, (k+1)*size, n -(k+1)*size )), 0)
    
    return val_input_set, val_target_set, train_input_set, train_target_set

def cross_validation(net, comp_train_input, comp_train_target, digit_train_input, digit_train_target, is_digit_model=False, use_aux_target=False, aux_loss_weight= None, fold = 4):
    
    """
    Perform a k-fold validation on a pytorch model, used to hyper-tune the learning rate.
    Print the mean validation error and std for each learning rate.
    
    Args:
        net (func): function returning pytorch neural net
        comp_train_input: training pairs
        comp_train_target: training labels
        digit_train_input: training digits
        digit_train_target: training classes
        is_digit_model (boolean): True if model trains on single digits, False if model trains on pairs of digits
        use_aux_target (boolean): True if model trains with an auxiliary loss
        aux_loss_weight (boolean): weight of auxiliary loss
        fold (int): number of folds
    """
    
    learning_rate = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    for lr in learning_rate:
        val_err = torch.empty((fold,1))
        for k in range(fold):
            model = net()
            comp_val_input_set, comp_val_target_set, comp_train_input_set, comp_train_target_set = split_dataset(comp_train_input, comp_train_target, k, fold)
            digit_val_input_set, digit_val_target_set, digit_train_input_set, digit_train_target_set = split_dataset(digit_train_input, digit_train_target, k, fold)
            
            aux_train_target = digit_train_target_set if use_aux_target else None       
            model_train_input = digit_train_input_set if is_digit_model else comp_train_input_set
            model_train_target = digit_train_target_set if is_digit_model else comp_train_target_set.float().view(-1,1)
            model_test_input = digit_val_input_set if is_digit_model else comp_val_input_set
            
            train_model(model, model_train_input, model_train_target, mini_batch_size=50, aux_train_target=aux_train_target, aux_loss_weight=aux_loss_weight, lr=lr)
            
            output = model(model_test_input)
            if use_aux_target:
                output = output[0]
        
            predicted = prediction_from_digit(output) if is_digit_model else prediction_from_comparaison(output)        
            
            
            error,_,_ = compute_nb_errors(predicted, comp_val_target_set)
            val_err[k] = error
        
        mean = torch.mean(val_err).item()
        std = torch.std(val_err).item()
        print("Lr : {:.6f} Mean Validation error : {:.6f} Std Validation error: {:.6f}".format(lr, mean, std))