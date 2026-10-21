import {
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  useId,
} from "react";
import "./r6-controls.css";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "quiet";
  pending?: boolean;
};

export function R6Button({
  variant = "secondary",
  pending = false,
  disabled,
  children,
  className = "",
  type = "button",
  ...props
}: ButtonProps) {
  return (
    <button
      {...props}
      type={type}
      className={`r6-button ${className}`}
      data-variant={variant}
      disabled={disabled || pending}
      aria-busy={pending || undefined}
    >
      <span className="r6-button-label">{children}</span>
      {pending && <span className="r6-button-spinner" aria-hidden="true" />}
    </button>
  );
}

export function R6Stepper({ active }: { active: 1 | 2 | 3 }) {
  return (
    <nav className="r6-stepper" aria-label="First-time setup progress">
      <ol>
        {["Import history", "Connect apps", "Distill memories"].map(
          (label, index) => (
            <li
              key={label}
              aria-current={active === index + 1 ? "step" : undefined}
            >
              <span aria-hidden="true">0{index + 1}</span> {label}
            </li>
          ),
        )}
      </ol>
    </nav>
  );
}

export function R6Switch({
  label,
  checked,
  pending = false,
  disabled = false,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  pending?: boolean;
  disabled?: boolean;
  onCheckedChange(checked: boolean): void;
}) {
  return (
    <button
      type="button"
      role="switch"
      className="r6-switch"
      aria-label={label}
      aria-checked={checked}
      aria-busy={pending || undefined}
      disabled={disabled}
      onClick={() => onCheckedChange(!checked)}
    >
      <span className="r6-switch-track" aria-hidden="true">
        <span className="r6-switch-thumb" />
      </span>
    </button>
  );
}

export function R6Field({
  label,
  error,
  id,
  className = "",
  "aria-describedby": describedBy,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & { label: string; error?: string }) {
  const generatedId = useId();
  const fieldId = id ?? generatedId;
  const errorId = `${fieldId}-error`;
  const description = [describedBy, error ? errorId : undefined]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={`r6-field ${className}`}>
      <label htmlFor={fieldId}>{label}</label>
      <input
        {...props}
        id={fieldId}
        aria-describedby={description || undefined}
        aria-invalid={error ? true : props["aria-invalid"]}
      />
      {error && (
        <p id={errorId} className="r6-field-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

export function R6LocalStorage() {
  return (
    <span
      className="r6-local-storage"
      title="Conversations and memories stay on this device; when a cloud model or an authorized assistant is used, the relevant content is sent to that service."
    >
      <span aria-hidden="true" />
      Local storage
    </span>
  );
}
