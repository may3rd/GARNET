import { jsPDF } from 'jspdf'
import type { DetectedObject, DetectionResult } from '@/types'
import { objectKey } from '@/lib/objectKey'

/**
 * Generates a PDF report for GARNET detection results.
 * Includes image preview, statistics, and object list table.
 */
export async function generatePdfReport(
    result: DetectionResult,
    reviewStatus: Record<string, 'accepted' | 'rejected'>,
    imageDataUrl: string
): Promise<Blob> {
    const doc = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
    })

    const pageWidth = doc.internal.pageSize.getWidth()
    const pageHeight = doc.internal.pageSize.getHeight()
    const margin = 15
    const contentWidth = pageWidth - margin * 2
    let y = margin

    // Title
    doc.setFontSize(20)
    doc.setFont('helvetica', 'bold')
    doc.text('GARNET Detection Report', margin, y)
    y += 10

    // Subtitle with date
    doc.setFontSize(10)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(100)
    const dateStr = new Date().toLocaleString()
    doc.text(`Generated: ${dateStr}`, margin, y)
    y += 8

    // Image filename
    const imageFileName = result.image_url.split('/').pop() || 'image.png'
    doc.text(`Source: ${imageFileName}`, margin, y)
    y += 12

    // Image preview
    if (imageDataUrl) {
        try {
            const maxImageWidth = contentWidth
            const maxImageHeight = 80

            // Calculate aspect ratio
            const aspectRatio = result.image_width / result.image_height
            let imgWidth = maxImageWidth
            let imgHeight = imgWidth / aspectRatio

            if (imgHeight > maxImageHeight) {
                imgHeight = maxImageHeight
                imgWidth = imgHeight * aspectRatio
            }

            doc.addImage(imageDataUrl, 'PNG', margin, y, imgWidth, imgHeight)
            y += imgHeight + 10
        } catch (error) {
            console.warn('Failed to add image to PDF:', error)
            y += 5
        }
    }

    // Statistics section
    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(0)
    doc.text('Summary Statistics', margin, y)
    y += 8

    const totalObjects = result.objects.length
    const acceptedCount = result.objects.filter(
        (obj) => reviewStatus[objectKey(obj)] === 'accepted'
    ).length
    const rejectedCount = result.objects.filter(
        (obj) => reviewStatus[objectKey(obj)] === 'rejected'
    ).length
    const pendingCount = totalObjects - acceptedCount - rejectedCount

    doc.setFontSize(10)
    doc.setFont('helvetica', 'normal')

    const stats = [
        ['Total Objects', totalObjects.toString()],
        ['Accepted', `${acceptedCount} (${totalObjects ? Math.round((acceptedCount / totalObjects) * 100) : 0}%)`],
        ['Rejected', `${rejectedCount} (${totalObjects ? Math.round((rejectedCount / totalObjects) * 100) : 0}%)`],
        ['Pending', `${pendingCount} (${totalObjects ? Math.round((pendingCount / totalObjects) * 100) : 0}%)`],
    ]

    stats.forEach(([label, value]) => {
        doc.text(`${label}:`, margin, y)
        doc.text(value, margin + 35, y)
        y += 5
    })
    y += 8

    // Category breakdown
    const categories = new Map<string, number>()
    result.objects.forEach((obj) => {
        const count = categories.get(obj.Object) || 0
        categories.set(obj.Object, count + 1)
    })

    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.text('Objects by Category', margin, y)
    y += 8

    doc.setFontSize(10)
    doc.setFont('helvetica', 'normal')

    const sortedCategories = Array.from(categories.entries()).sort((a, b) => b[1] - a[1])
    sortedCategories.forEach(([category, count]) => {
        const displayName = category.replace(/_/g, ' ')
        doc.text(`• ${displayName}: ${count}`, margin, y)
        y += 5
        if (y > pageHeight - 30) {
            doc.addPage()
            y = margin
        }
    })
    y += 8

    // Object list table
    doc.addPage()
    y = margin

    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.text('Detection Results', margin, y)
    y += 10

    // Table header
    doc.setFontSize(9)
    doc.setFont('helvetica', 'bold')
    doc.setFillColor(240, 240, 240)
    doc.rect(margin, y - 4, contentWidth, 7, 'F')

    const colWidths = [15, 45, 25, 25, 50]
    const cols = ['#', 'Class', 'Confidence', 'Status', 'OCR Text']
    let x = margin + 2
    cols.forEach((col, i) => {
        doc.text(col, x, y)
        x += colWidths[i]
    })
    y += 6

    // Table rows
    doc.setFont('helvetica', 'normal')
    result.objects.forEach((obj, index) => {
        if (y > pageHeight - 15) {
            doc.addPage()
            y = margin

            // Repeat header on new page
            doc.setFont('helvetica', 'bold')
            doc.setFillColor(240, 240, 240)
            doc.rect(margin, y - 4, contentWidth, 7, 'F')
            x = margin + 2
            cols.forEach((col, i) => {
                doc.text(col, x, y)
                x += colWidths[i]
            })
            y += 6
            doc.setFont('helvetica', 'normal')
        }

        const status = reviewStatus[objectKey(obj)]
        const statusLabel = status === 'accepted' ? '✓ Accepted' : status === 'rejected' ? '✗ Rejected' : 'Pending'
        const row = [
            (index + 1).toString(),
            obj.Object.replace(/_/g, ' ').slice(0, 20),
            `${Math.round(obj.Score * 100)}%`,
            statusLabel,
            (obj.Text || '—').slice(0, 25),
        ]

        x = margin + 2
        row.forEach((cell, i) => {
            doc.text(cell, x, y)
            x += colWidths[i]
        })
        y += 5
    })

    return doc.output('blob')
}

/**
 * Converts an image URL to a data URL for embedding in PDF.
 */
export async function getImageAsDataUrl(imageUrl: string): Promise<string> {
    return new Promise((resolve, reject) => {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => {
            const canvas = document.createElement('canvas')
            canvas.width = img.naturalWidth
            canvas.height = img.naturalHeight
            const ctx = canvas.getContext('2d')
            if (!ctx) {
                reject(new Error('Failed to get canvas context'))
                return
            }
            ctx.drawImage(img, 0, 0)
            resolve(canvas.toDataURL('image/png'))
        }
        img.onerror = () => reject(new Error('Failed to load image'))
        img.src = imageUrl
    })
}
