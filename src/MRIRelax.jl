
module MRIRelax

	export longitudinal, transverse
	export fit_inversion_recovery, inversion_recovery_model
	export fit_transverse

	using LinearAlgebra
	using LsqFit

	@inline function longitudinal(t::Real, R1::Real, M::Real)::Real
		1.0 - (1.0 - M) * exp(-R1 * t)
	end
	@inline function transverse(t::Real, R2::Real)::Real
		exp(-R2 * t)
	end

	@inline function inversion_recovery_model(
		signal::AbstractVector{<: Real},
		t::AbstractVector{<: Real},
		(R1, Minv, M0)::AbstractVector{<: Real}
	)::Vector{Real}
		@. signal = M0 * abs(longitudinal(t, R1, Minv))
	end

	"""
		TODO: Two pool models?

	"""
	function fit_inversion_recovery(
		Tinv::AbstractVector{<: Real},
		signal::AbstractVector{<: Real},
		Δsignal::AbstractVector{<: Real},
		T1::Real,
		Minv::Real,
		M0::Real
	)::NTuple{6, Real}

		NaNs = ntuple(i -> NaN,  Val(6))

		result = curve_fit(
			inversion_recovery_model,
			Tinv,
			signal,
			1.0 ./ Δsignal.^2,
			[1.0 / T1, Minv, M0]; # Initial guess
			autodiff=:forwarddiff,
			inplace=true
		)
		!result.converged && return NaNs

		# Get parameters
		R1, Minv, M0 = coef(result)

		# Compute std-errors
		ΔR1, ΔMinv, ΔM0 = let
			J = result.jacobian
			Q, R = qr(J)
			det(R) == 0 && return NaNs
			Rinv = inv(R)
			covar = Rinv * Rinv'
			var = diag(covar)
			any(var .< 0.0) && return NaNs
			sqrt.(var)
		end

		# Get T1 from R1
		T1 = 1.0 / R1
		ΔT1 = T1^2 * ΔR1

		return T1, ΔT1, Minv, ΔMinv, M0, ΔM0
	end


	"""
		fit_transverse(TE, signal)

	Find the transverse relaxation time.

	Fits an exponential decay to the absolute of the transverse magnetisation `signal`
	measured at echo times `TE`.

	The exponential decay
		
		s(t) = M0 * exp(-R2 * t)
		
	with R2 = 1/T2, can be rewritten as
		
		log(s(t)) = log(M0) - R2 * t

	making it linear in log(M0) and R2.

	Errors for the fit are computed via Gaussian error propagation
		
		Δ(log(s(t))) = |∂log(s)/∂s| * Δs =  1/s * Δs

	where Δs is assumed constant. Note that s >=0 (absolute!).

	Errors of the parameter estimates are computed using the same technique
		
		ΔT2 = |∂T2/∂R2| * ΔR2 = T2² * ΔR2 

		ΔM0 = |∂M0/∂log(M0)| * ΔM0 = M0 * Δlog(M0)

	signal must be > 0
	"""
	function fit_transverse(
		TE::AbstractVector{<: Real},
		signal::AbstractVector{<: Real},
		Δsignal::AbstractVector{<: Real}
	)::NTuple{4, Real}

		c, Δc = linreg(
			[ TE ones(length(TE)) ], # Use TE and constant term, the latter fitting for M0
			log.(signal),
			signal ./ Δsignal # 1 / Propagated error of log.(signal)
		)
		isnan(c[1]) && return NaN, NaN, NaN, NaN

		R2 = -c[1]
		T2 = 1.0 / R2
		ΔT2 = T2^2 * Δc[1]

		M0 = exp(c[2])
		ΔM0 = M0 * Δc[2]

		return T2, ΔT2, M0, ΔM0
	end
	function linreg( # TODO: outsource
		X::Matrix{T},
		y::Vector{T},
		w::Vector{T}
	)::Tuple{Vector{T}, Vector{T}} where T <: Union{Float64, ComplexF64}
		# Make note about upper case variables
		W = diagm(w.^2)
		A = X' * W
		H = A * X
		if det(H) == 0
			θ = Vector{T}(undef, size(X, 1))
			fill!(θ, NaN)
			Δθ = θ
		else
			θ = H \ (A * y)
			Δθ = sqrt.(diag(inv(H)))
		end
		return θ, Δθ
	end

end

