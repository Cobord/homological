use super::field_generals::Ring;

#[allow(clippy::module_name_repetitions)]
pub trait BezoutDomain: Ring {
    /// division
    /// # Errors
    /// divide by 0
    #[allow(clippy::result_unit_err)]
    fn divide_by_divisor(&self, other: &Self) -> Result<Self, ()>;

    /// greatest common divisor in this Bezout Domain
    #[must_use]
    fn gcd(&self, other: &Self) -> Self;

    /// `sigma*self + tau*other = beta`
    /// don't demand division by `beta` unless caller needs it
    /// `alpha \equiv self/beta`
    /// `gamma \equiv other/beta`
    /// `sigma*alpha + tau*gamma = beta/beta = 1`
    /// the format of the output is `[sigma,tau],beta`
    /// don't give `alpha` or `gamma`, let caller produce those if needed
    fn gcd_and_witnesses(&self, other: &Self) -> ([Self; 2], Self);

    /// see articles on Bezout Domains for `L_0`
    fn l0_matrix(&self, other: &Self) -> [[Self; 2]; 2] {
        let ([sigma, tau], beta) = self.gcd_and_witnesses(other);
        let alpha = self
            .divide_by_divisor(&beta)
            .expect("Dividing by the GCD which is a divisor");
        let gamma = other
            .divide_by_divisor(&beta)
            .expect("Dividing by the GCD which is a divisor");
        [[sigma, tau], [-gamma, alpha]]
    }

    /// see articles on Bezout Domains for `L_0`
    fn l0_matrix_inverse(&self, other: &Self) -> [[Self; 2]; 2] {
        let ([sigma, tau], beta) = self.gcd_and_witnesses(other);
        let alpha = self
            .divide_by_divisor(&beta)
            .expect("Dividing by the GCD which is a divisor");
        let gamma = other
            .divide_by_divisor(&beta)
            .expect("Dividing by the GCD which is a divisor");
        [[alpha, -tau], [gamma, sigma]]
    }
}
