import torch

def testing_data(model , device , test_loader , criterion):
    
    threshold = 0.5

    correct_frame_correct_pre = 0
    correct_frame_error_pre   = 0
    error_frame_error_pre     = 0
    error_frame_correct_pre   = 0

    correct_pre = 0;
    error_pre   = 0; 

    model.eval()
    test_loss = 0

    for batch_index , (data,label) in enumerate(test_loader):
        data , label = data.to(device) , label.to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output,label)

        test_loss += loss.item() * len(data)  # loss * batch 
        average_loss = test_loss / len(test_loader.dataset)

        correct_frame_correct_pre += ((output< threshold) & (label==0)).sum().item()
        correct_frame_error_pre   += ((output>=threshold) & (label==0)).sum().item()
        error_frame_error_pre     += ((output>=threshold) & (label==1)).sum().item()
        error_frame_correct_pre   += ((output< threshold) & (label==1)).sum().item()

        correct_pre += ((output< threshold) & (label==0)).sum().item()
        correct_pre += ((output>=threshold) & (label==1)).sum().item()
        error_pre   += ((output>=threshold) & (label==0)).sum().item()
        error_pre   += ((output< threshold) & (label==1)).sum().item()

    print('\tnumber of correct predict =',correct_pre)
    print('\tnumber of error predict   =',error_pre)
    print('\taverage loss :{:.4f}'.format(average_loss))
    print('\tcorrect_frame_correct_pre =',correct_frame_correct_pre)    # FN
    print('\tcorrect_frame_error_pre   =',correct_frame_error_pre)      # FP
    print('\terror_frame_error_pre     =',error_frame_error_pre)        # TP
    print('\terror_frame_correct_pre   =',error_frame_correct_pre)      # TN

    TPR = error_frame_error_pre/(error_frame_error_pre+error_frame_correct_pre)
    FPR = correct_frame_error_pre/(correct_frame_correct_pre+correct_frame_error_pre)
    accurary = correct_pre/(correct_pre+error_pre)
    Recall = error_frame_error_pre/(error_frame_error_pre+error_frame_correct_pre)
    Precision = error_frame_error_pre/(error_frame_error_pre+correct_frame_error_pre)
    F1_score = 2 * Precision * Recall / (Precision + Recall)

    print('\tTPR =',TPR)
    print('\tFPR =',FPR)
    print('\taccuracy =',accurary)
    print('\tRecall =',Recall)
    print('\tPrecision =',Precision)
    print('\tF1_score =',F1_score)

    
    return TPR,FPR,accurary