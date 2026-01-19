def feature_and_index_all_batch(val_loader, model_device, model,  label_index = None, no_feature0 = True, one_batch = False):
    '''
    OA, F1, miou, precision, recall, feature_source_mean, feature_source_var
    '''
    index_list = []
    num_image = 0
    feature_mean_batch_list = []
    extracted_features = []

    def one_batch_iter():
        images, true_masks_cpu = next(val_loader)
        # count the number of images
        num_image += images.shape[0]
        images = images.to(device=model_device, dtype=torch.float32)
        true_masks = true_masks_cpu.to(device=model_device, dtype=torch.long)
        with torch.no_grad():
            model_output = model(images)
        predictions = torch.argmax(model_output, dim=1)
        feature_source = feature_from_hook['hook_output'].cpu()
        # extract the feature of label_index
        batch_features = feature_source.permute(0, 2, 3, 1)
        if label_index is not None:
            batch_labels = true_masks_cpu.unsqueeze(1).expand_as(feature_source).permute(0, 2, 3, 1)
            extracted_batch_features = batch_features[batch_labels == label_index].reshape(-1, feature_source.shape[1])
        else:
            extracted_batch_features = batch_features.reshape(-1, feature_source.shape[1])
        extracted_features.extend(extracted_batch_features.tolist())
        
        
        OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2, ignore_label_0=False if label_index is None else True)
        index_list.append([OA, F1, miou, precision, recall])
        # feature_mean_batch_list.append(torch.mean(feature_source, dim=[0,2,3]).flatten().cpu())
    
    if one_batch:
        one_batch_iter()
    else:
        for i in tqdm(range(len(val_loader)), desc="features_all_batch"):
            one_batch_iter()
            # if num_image > 150:
            if num_image > 100:
                break

    extracted_features_tensor = torch.tensor(extracted_features, dtype=torch.float32)
    # no_feature0 = True
    if no_feature0:
        feature_mean = torch.zeros(extracted_features_tensor.shape[1], dtype=torch.float32)
        feature_var = torch.zeros(extracted_features_tensor.shape[1], dtype=torch.float32)
        for i in range(extracted_features_tensor.shape[1]):
            feature_mask = extracted_features_tensor[:, i] != 0.0
            masked_feature = extracted_features_tensor[feature_mask, i]
            feature_mean[i] = torch.mean(masked_feature)
            feature_var[i] = torch.var(masked_feature)
    else:
        feature_mean = torch.mean(extracted_features_tensor, dim=0)
        feature_var = torch.var(extracted_features_tensor, dim=0)
    
    

    # feature_var = feature_mean
    OA_all, F1_all, miou_all, precision_all, recall_all = torch.tensor(index_list).mean(dim=0).tolist()
    return OA_all, F1_all, miou_all, precision_all, recall_all, feature_mean, feature_var, np.array(extracted_features, dtype=np.float32)