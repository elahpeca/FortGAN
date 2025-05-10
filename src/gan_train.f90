module gan_train
    use nf_network, only: network
    use nf_optimizers, only: optimizer_base_type
    use nf_random, only: random_normal
    implicit none

    integer, private :: step_counter = 0

contains
    subroutine train_gan_step( &
        gen, &
        disc, & 
        real_images, &
        batch_size, &
        noise_dim, &
        gen_optimizer, &
        disc_optimizer &
    )
    class(network), intent(inout) :: gen
    class(network), intent(inout) :: disc
    real, intent(in) :: real_images(:,:)  
    integer, intent(in) :: batch_size, noise_dim
    class(optimizer_base_type), intent(in) :: gen_optimizer, disc_optimizer
    
    real, allocatable :: noise(:,:), fake_images(:,:)
    real, allocatable :: real_labels(:,:), fake_labels(:,:)  
    real, allocatable :: gen_labels(:,:)                     

    step_counter = step_counter + 1

    allocate(real_labels(1, batch_size), source=1.0)
    allocate(fake_labels(1, batch_size), source=0.0)
    allocate(gen_labels(1, batch_size), source=1.0)
    
    allocate(noise(noise_dim, batch_size))
    call random_normal(noise)
    
    print *, "Step:", step_counter

    ! train discriminator on real data
    call disc%set_training_mode(.true.)

    call disc%train( &
        input_data = real_images, &       
        output_data = real_labels, &       
        batch_size = batch_size, &
        epochs = 1, &
        optimizer = disc_optimizer &
    )
    
    ! train discriminator on fake data
    fake_images = gen%predict_batch(noise) 
    call disc%train( &
        input_data = fake_images, &        
        output_data = fake_labels, &      
        batch_size = batch_size, &
        epochs = 1, &
        optimizer = disc_optimizer &
    )
    
    ! GENERATOR TRAIN LOGIC (COMING SOON)

    end subroutine train_gan_step
end module gan_train