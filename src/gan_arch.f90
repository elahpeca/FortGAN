module gan_arch
    use nf, only: network
    implicit none

    type :: gan
        type(network), pointer :: generator
        type(network), pointer :: discriminator
    contains
        procedure :: init => init_gan
        procedure :: destroy => destroy_gan
        procedure :: get_generator => get_gan_generator  
        procedure :: get_discriminator => get_gan_discriminator 
    end type gan

    contains

    subroutine init_gan(this, generator_impl, discriminator_impl)
        class(gan), intent(inout) :: this
        class(network), intent(in) :: generator_impl, discriminator_impl 
        
        allocate(this%generator, source = generator_impl)
        allocate(this%discriminator, source = discriminator_impl)

    end subroutine init_gan

    subroutine destroy_gan(this)
        class(gan), intent(inout) :: this

        if (associated(this%generator)) then
            deallocate(this%generator)
            nullify(this%generator)
        end if

        if (associated(this%discriminator)) then
            deallocate(this%discriminator)
            nullify(this%discriminator)
        end if
        
    end subroutine destroy_gan

    ! getters for generator and discriminator 
    function get_gan_generator(this) result(gen)
        class(gan), intent(in) :: this
        type(network), pointer :: gen
        gen => this%generator

    end function get_gan_generator

    function get_gan_discriminator(this) result(disc)
        class(gan), intent(in) :: this
        type(network), pointer :: disc
        disc => this%discriminator
        
    end function get_gan_discriminator

end module gan_arch

