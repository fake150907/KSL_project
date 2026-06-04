import { useState } from 'react'

type SignLanguageLogoProps = {
  className?: string
  size?: number
}

export default function SignLanguageLogo({ className = '', size }: SignLanguageLogoProps) {
  const [imageLoaded, setImageLoaded] = useState(true)

  return (
    <div
      className={`relative flex shrink-0 items-center justify-center overflow-hidden rounded-[22%] bg-blue-600 shadow-xl shadow-blue-100 ${className}`}
      style={{
        width: size,
        height: size,
      }}
    >
      <div className="absolute inset-0 bg-[linear-gradient(135deg,#4f7df7_0%,#2451dc_58%,#2fc8d2_100%)]" />
      {imageLoaded && (
        <img
          src="/sign-language-logo.png"
          alt=""
          aria-hidden="true"
          className="relative h-full w-full translate-y-[4%] object-contain"
          draggable={false}
          onError={() => setImageLoaded(false)}
        />
      )}
    </div>
  )
}
