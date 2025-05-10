module gan_arch
    use nf, only: network
    implicit none

    type :: gan
        type(network), pointer :: generator
        type(network), pointer :: discriminator
    contains
        procedure :: init => init_gan
    end type gan

    contains

    subroutine init_gan(this, generator_impl, discriminator_impl)
        class(GAN), intent(inout) :: this
        class(network), intent(in) :: generator_impl, discriminator_impl 
        
        allocate(this%generator, source = generator_impl)
        allocate(this%discriminator, source = discriminator_impl)

    end subroutine init_gan

end module gan_arch

