program main
    use gan_arch, only: gan
    use gan_factories, only: create_generator, create_discriminator
    use data_utils, only: load_data
    use gan_train_single, only: train_gan_single 
    use nf, only: network, adam, mse
    implicit none

    type(gan) :: g
    real, allocatable :: training_images(:,:)
    real, allocatable :: validation_images(:,:)
    real, allocatable :: testing_images(:,:)
    real, allocatable :: params(:)
    integer :: batch_size, noise_dim, epochs
    integer :: i, j 
    real :: gen_lr = 0.0002, disc_lr = 0.0002
    type(adam) :: optimizer_gen
    type(adam) :: optimizer_disc
    type(mse) :: generator_loss_fn, discriminator_loss_fn

    noise_dim = 100
    epochs = 10

    call g%init( &
        create_generator(noise_dim=noise_dim, img_size=784), &
        create_discriminator(img_size=784) &
    )

    call load_data(training_images, validation_images, testing_images, normalize=.true.)
    print *, "Loaded ", size(training_images,2), " training images"
    print *, "Image dimensions: ", size(training_images,1)
    print *, "Batch size (unused for single image training): ", batch_size
    print *, "Total images per epoch: ", size(training_images,2)

    optimizer_gen = adam(learning_rate=gen_lr, beta1=0.5)
    optimizer_disc = adam(learning_rate=disc_lr, beta1=0.5)

    do i = 1, epochs
        print *, "Epoch:", i
        do j = 1, size(training_images, 2) ! single image iteration (temporarily)
            call train_gan_single( &
                gan_instance=g, &
                real_image=training_images(:, j), &
                noise_dim=noise_dim, &
                optimizer_gen=optimizer_gen, &
                optimizer_disc=optimizer_disc, &
                loss_disc=discriminator_loss_fn, &
                loss_gen=generator_loss_fn &
            )
            if (mod(j, 100) == 0) then
                print *, "  Processed image:", j, "/", size(training_images, 2)
            end if
        end do
        print *, "Completed Epoch:", i
        print *, "----------------------------------------"
    end do

    call g%destroy()
    print *, "Training completed successfully."

end program main