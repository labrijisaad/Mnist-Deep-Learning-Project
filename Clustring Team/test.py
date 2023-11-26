# Selecting the model
classifier1_augmented, classifier2_augmented = model1, model1

# Compile the classifier1_augmented and classifier2_augmented
classifier1_augmented.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier2_augmented.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the number of elements used to train the first classifier
print(f">>>> Number of elements used to train the first classifier: {len(full_x_labeled)}")

# Train the model with the original labeled data and validation set, and save the history
history_classifier1_augmented = classifier1_augmented.fit(
    full_x_labeled,  # x_labeled and augmented_x_labeled
    full_y_labeled,  # y_labeled and augmented_y_labeled
    epochs=30,
    batch_size=32,
    validation_data=(x_validation, y_validation)
)

# Pseudo-labeling
pseudo_labeled_images, pseudo_labels, x_unlabeled_unused = pseudo_labeling(classifier1_augmented, x_unlabeled)

if pseudo_labeled_images is not None:
    # Update labeled data with pseudo-labeled data
    x_labeled_and_pseudo_labeled_images = np.concatenate([full_x_labeled, pseudo_labeled_images])
    y_labeled_and_pseudo_labeled_images = np.concatenate([full_y_labeled, pseudo_labels])

    # Print the number of elements used to train the first classifier and pseudo-labeled data
    print(
        f">>>> Number of elements used to train the second classifier : {len(x_labeled_and_pseudo_labeled_images) + len(x_unlabeled_unused)} ( number of augmented elements: {len(augmented_x_labeled)}")

    # Train the classifier2 on the full dataset (original + pseudo-labeled) with validation set, and save the history
    history_classifier2_augmented = classifier2_augmented.fit(
        np.concatenate([x_labeled_and_pseudo_labeled_images, x_unlabeled_unused]),
        np.concatenate([y_labeled_and_pseudo_labeled_images,
                        np.argmax(classifier2_augmented.predict(x_unlabeled_unused), axis=1)]),
        epochs=5,
        batch_size=8,
        validation_data=(x_validation, y_validation)
    )

# Evaluate the classifier2 Performance on Test Data
test_loss, test_acc = classifier2.evaluate(x_test, y_test)
print(f"classifier2 - Test Loss: {test_loss}, Test Accuracy: {test_acc}\n")
