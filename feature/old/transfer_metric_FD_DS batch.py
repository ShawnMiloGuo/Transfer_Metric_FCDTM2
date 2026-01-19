def feature_and_index_batch(val_loader, model_device, model, label_index = None, no_feature0 = True):
    '''
    OA, F1, miou, precision, recall, feature_source_mean, feature_source_var
    '''
        images, true_masks_cpu = next(val_loader)
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
    
    # no_feature0 = True
    if no_feature0:
        feature_mean = torch.zeros(extracted_batch_features.shape[1], dtype=torch.float32)
        feature_var = torch.zeros(extracted_batch_features.shape[1], dtype=torch.float32)
        for i in range(extracted_batch_features.shape[1]):
            feature_mask = extracted_batch_features[:, i] != 0.0
            masked_feature = extracted_batch_features[feature_mask, i]
            feature_mean[i] = torch.mean(masked_feature)
            feature_var[i] = torch.var(masked_feature)
    else:
        feature_mean = torch.mean(extracted_batch_features, dim=0)
        feature_var = torch.var(extracted_batch_features, dim=0)
    
    OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2, ignore_label_0=False if label_index is None else True)
    return OA_all, F1_all, miou_all, precision_all, recall_all, feature_mean, feature_var, np.array(extracted_features, dtype=np.float32)