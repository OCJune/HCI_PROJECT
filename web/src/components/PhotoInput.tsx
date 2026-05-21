import { Upload } from 'lucide-react'
import { useRef } from 'react'

interface PhotoInputProps {
  image?: string | null
  setImage: (image: string | null) => void
}

const PhotoInput = ({ image, setImage }: PhotoInputProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  return (
    <div
      className="relative flex aspect-4/3 w-full max-w-4xl cursor-pointer items-center justify-center overflow-hidden rounded-2xl border-4 border-dashed border-gray-100 bg-white shadow-xl transition-all hover:ring-4 hover:ring-blue-500/30"
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
        <div className="flex flex-col items-center gap-4 px-6 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white shadow-md">
            <Upload size={32} className="text-blue-600" />
          </div>
          <div>
            <p className="text-xl font-bold text-gray-900">사진 업로드하기</p>
            <p className="mt-1 text-sm text-gray-500">
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
