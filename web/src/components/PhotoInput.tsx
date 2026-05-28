import { Upload } from 'lucide-react'
import { useRef } from 'react'

interface PhotoInputProps {
  image?: string | null
  setImage: (image: string | null) => void
  setFile: (file: File | null) => void
}

const PhotoInput = ({ image, setImage, setFile }: PhotoInputProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    } else {
      setFile(null)
      setImage(null)
    }
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  return (
    <div
      className="relative flex aspect-4/3 w-full max-w-4xl cursor-pointer items-center justify-center overflow-hidden rounded-2xl border-4 border-dashed border-gray-100 bg-white transition-all hover:ring-4 hover:ring-blue-500/30"
      onClick={triggerFileInput}
    >
      {image ? (
        <div className="h-full w-full bg-white">
          <img
            src={image}
            alt="Uploaded"
            className="h-full w-full object-contain"
          />
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 px-6 text-center md:gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white shadow-md md:h-16 md:w-16">
            <Upload className="h-6 w-6 text-blue-600 md:h-8 md:w-8" />
          </div>
          <div>
            <p className="text-lg font-bold text-gray-900 md:text-xl">사진 업로드하기</p>
            <p className="mt-1 text-xs text-gray-500 md:text-sm">
              클릭하여 컬러링북으로 만들 사진을 선택하세요
            </p>
          </div>
        </div>
      )}
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        accept="image/*"
        onChange={handleImageUpload}
      />
    </div>
  )
}

export default PhotoInput
